# Import superglue
from superglue.models.superpoint import SuperPoint
from superglue.models.superglue import SuperGlue
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
import os
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm
from PIL import Image
import numpy as np
import h5py
from fastprogress import progress_bar
import pycolmap
from collections import defaultdict
from time import time
from copy import deepcopy



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SuperGlueCustomMatchingV2(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}, device=None):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

        self.tta_map = {
            'orig': self.untta_none,
            'eqhist': self.untta_none,
            'clahe': self.untta_none,
            'flip_lr': self.untta_fliplr,
            'flip_ud': self.untta_flipud,
            'rot_r10': self.untta_rotr10,
            'rot_l10': self.untta_rotl10,
            'fliplr_rotr10': self.untta_fliplr_rotr10,
            'fliplr_rotl10': self.untta_fliplr_rotl10
        }
        self.device = device

    def forward_flat(self, data, ttas=['orig', ], tta_groups=[['orig']]):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        # sp_st = time.time()
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        # sp_nd = time.time()
        # print('SP:', sp_nd - sp_st, 's')

        # Reverse-tta before inference
        pred['scores0'] = list(pred['scores0'])
        pred['scores1'] = list(pred['scores1'])
        for i in range(len(pred['keypoints0'])):
            pred['keypoints0'][i], pred['descriptors0'][i], pred['scores0'][i] = self.tta_map[ttas[i]](
                pred['keypoints0'][i], pred['descriptors0'][i], pred['scores0'][i],
                w=data['image0'].shape[3], h=data['image0'].shape[2], inplace=True, mask_illegal=True)

            pred['keypoints1'][i], pred['descriptors1'][i], pred['scores1'][i] = self.tta_map[ttas[i]](
                pred['keypoints1'][i], pred['descriptors1'][i], pred['scores1'][i],
                w=data['image1'].shape[3], h=data['image1'].shape[2], inplace=True, mask_illegal=True)

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        group_preds = []
        for tta_group in tta_groups:
            group_mask = torch.from_numpy(np.array([x in tta_group for x in ttas], dtype=np.bool))
            group_data = {
                **{f'keypoints{k}': [data[f'keypoints{k}'][i] for i in range(len(ttas)) if ttas[i] in tta_group] for k in [0, 1]},
                **{f'descriptors{k}': [data[f'descriptors{k}'][i] for i in range(len(ttas)) if ttas[i] in tta_group] for k in [0, 1]},
                **{f'scores{k}': [data[f'scores{k}'][i] for i in range(len(ttas)) if ttas[i] in tta_group] for k in [0, 1]},
                **{f'image{k}': data[f'image{k}'][group_mask, ...] for k in [0, 1]},
            }
            for k, v in group_data.items():
                if isinstance(group_data[k], (list, tuple)):
                    if k.startswith('descriptor'):
                        group_data[k] = torch.cat(group_data[k], 1)[None, ...]
                    else:
                        group_data[k] = torch.cat(group_data[k])[None, ...]
                else:
                    group_data[k] = torch.flatten(group_data[k], 0, 1)[None, ...]
            # sg_st = time.time()
            group_pred = {
                # **{k: group_data[k] for k in group_data},
                **group_data,
                **self.superglue(group_data)
            }
            # sg_nd = time.time()
            # print('SG:', sg_nd - sg_st, 's')
            group_preds.append(group_pred)
        return group_preds

    def forward_cross(self, data, ttas=['orig', ], tta_groups=[('orig', 'orig')]):
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        sp_st = time()
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        sp_nd = time()

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        # Group predictions (list, with elements with matches{0,1}, matching_scores{0,1} keys)
        group_pred_list = []
        tta2id = {k: i for i, k in enumerate(ttas)}
        for tta_group in tta_groups:
            group_idx = tta2id[tta_group[0]], tta2id[tta_group[1]]
            group_data = {
                **{f'image{i}': data[f'image{i}'][group_idx[i]:group_idx[i]+1] for i in [0, 1]},
                **{f'keypoints{i}': data[f'keypoints{i}'][group_idx[i]:group_idx[i]+1] for i in [0, 1]},
                **{f'descriptors{i}': data[f'descriptors{i}'][group_idx[i]:group_idx[i]+1] for i in [0, 1]},
                **{f'scores{i}': data[f'scores{i}'][group_idx[i]:group_idx[i]+1] for i in [0, 1]},
            }

            for k in group_data:
                if isinstance(group_data[k], (list, tuple)):
                    group_data[k] = torch.stack(group_data[k])

            group_sg_pred = self.superglue(group_data)
            group_pred_list.append(group_sg_pred)

        # UnTTA
        data['scores0'] = list(data['scores0'])
        data['scores1'] = list(data['scores1'])
        for i in range(len(data['keypoints0'])):
            data['keypoints0'][i], data['descriptors0'][i], data['scores0'][i] = self.tta_map[ttas[i]](
                data['keypoints0'][i], data['descriptors0'][i], data['scores0'][i],
                w=data['image0'].shape[3], h=data['image0'].shape[2], inplace=True, mask_illegal=False)

            data['keypoints1'][i], data['descriptors1'][i], data['scores1'][i] = self.tta_map[ttas[i]](
                data['keypoints1'][i], data['descriptors1'][i], data['scores1'][i],
                w=data['image1'].shape[3], h=data['image1'].shape[2], inplace=True, mask_illegal=False)

        # Sooo... groups?
        for group_pred, tta_group in zip(group_pred_list, tta_groups):
            group_idx = tta2id[tta_group[0]], tta2id[tta_group[1]]
            group_pred.update({
                **{f'keypoints{i}': data[f'keypoints{i}'][group_idx[i]:group_idx[i]+1] for i in [0, 1]},
                **{f'scores{i}': data[f'scores{i}'][group_idx[i]:group_idx[i]+1] for i in [0, 1]},
            })
        return group_pred_list


    def untta_none(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        if not inplace:
            keypoints = keypoints.clone()
        return keypoints, descriptors, scores
    
    def untta_fliplr(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        if not inplace:
            keypoints = keypoints.clone()
        keypoints[:, 0] = w - keypoints[:, 0] - 1.
        return keypoints, descriptors, scores

    def untta_flipud(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        if not inplace:
            keypoints = keypoints.clone()
        keypoints[:, 1] = h - keypoints[:, 1] - 1.
        return keypoints, descriptors, scores

    def untta_rotr10(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        # rotr10 is +10, inverse is -10
        rot_M_inv = torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), -15, 1)).to(torch.float32).to(self.device)
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        if mask_illegal:
            mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < w) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < h)
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores

    def untta_rotl10(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        # rotr10 is -10, inverse is +10
        rot_M_inv = torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)).to(torch.float32).to(self.device)
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        if mask_illegal:
            mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < w) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < h)
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores
        
    def untta_fliplr_rotr10(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        # rotr10 is +10, inverse is -10
        rot_M_inv = torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), -15, 1)).to(torch.float32).to(self.device)
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        rot_kpts[:, 0] = w - rot_kpts[:, 0] - 1.
        if mask_illegal:
            mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < w) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < h)
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores

    def untta_fliplr_rotl10(self, keypoints, descriptors, scores, w, h, inplace=True, mask_illegal=True):
        # rotr10 is -10, inverse is +10
        rot_M_inv = torch.from_numpy(cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1)).to(torch.float32).to(self.device)
        ones = torch.ones_like(keypoints[:, 0])
        hom = torch.cat([keypoints, ones[:, None]], 1)
        rot_kpts = torch.matmul(rot_M_inv, hom.T).T[:, :2]
        rot_kpts[:, 0] = w - rot_kpts[:, 0] - 1.
        if mask_illegal:
            mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < w) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < h)
            return rot_kpts[mask], descriptors[:, mask], scores[mask]
        else:
            return rot_kpts, descriptors, scores


def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices

class SuperGlueMatcherV2:
    def __init__(self, config, device=None, conf_th=None):
        self.config = config
        self.device = device
        self._superglue_matcher = SuperGlueCustomMatchingV2(
            config=config, device=self.device,
            ).eval().to(device)
        self.conf_thresh = conf_th
    
    def prep_np_img(self, img, long_side=None):
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1])
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        else:
            scale = 1.0
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scale
    
    def frame2tensor(self, frame):
        return (torch.from_numpy(frame).float()/255.)[None, None].to(self.device)
            
    def tta_rotation_preprocess(self, img_np, angle):
        rot_M = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1)
        rot_M_inv = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1)
        rot_img = self.frame2tensor(cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0])))
        return rot_M, rot_img, rot_M_inv

    def tta_rotation_postprocess(self, kpts, img_np, rot_M_inv):
        ones = np.ones(shape=(kpts.shape[0], ), dtype=np.float32)[:, None]
        hom = np.concatenate([kpts, ones], 1)
        rot_kpts = rot_M_inv.dot(hom.T).T[:, :2]
        mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < img_np.shape[1]) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < img_np.shape[0])
        return rot_kpts, mask
# 
    def __call__(self, img_np0, img_np1, input_longside, tta_groups=[('orig', 'orig')], forward_type='cross'):
        with torch.no_grad():
            img_np0, scale0 = self.prep_np_img(img_np0, input_longside)
            img_np1, scale1 = self.prep_np_img(img_np1, input_longside)

            img_ts0 = self.frame2tensor(img_np0)
            img_ts1 = self.frame2tensor(img_np1)
            images0, images1 = [], []

            tta = []
            for tta_g in tta_groups:
                tta += tta_g
            tta = list(set(tta))

            # TTA
            for tta_elem in tta:
                if tta_elem == 'orig':
                    img_ts0_aug, img_ts1_aug = img_ts0, img_ts1
                elif tta_elem == 'flip_lr':
                    img_ts0_aug = torch.flip(img_ts0, [3, ])
                    img_ts1_aug = torch.flip(img_ts1, [3, ])
                elif tta_elem == 'flip_ud':
                    img_ts0_aug = torch.flip(img_ts0, [2, ])
                    img_ts1_aug = torch.flip(img_ts1, [2, ])
                elif tta_elem == 'rot_r10':
                    rot_r10_M0, img_ts0_aug, rot_r10_M0_inv = self.tta_rotation_preprocess(img_np0, 15)
                    rot_r10_M1, img_ts1_aug, rot_r10_M1_inv = self.tta_rotation_preprocess(img_np1, 15)
                elif tta_elem == 'rot_l10':
                    rot_l10_M0, img_ts0_aug, rot_l10_M0_inv = self.tta_rotation_preprocess(img_np0, -15)
                    rot_l10_M1, img_ts1_aug, rot_l10_M1_inv = self.tta_rotation_preprocess(img_np1, -15)
                elif tta_elem == 'fliplr_rotr10':
                    rot_r10_M0, img_ts0_aug, rot_r10_M0_inv = self.tta_rotation_preprocess(img_np0[:, ::-1], 15)
                    rot_r10_M1, img_ts1_aug, rot_r10_M1_inv = self.tta_rotation_preprocess(img_np1[:, ::-1], 15)
                elif tta_elem == 'fliplr_rotl10':
                    rot_l10_M0, img_ts0_aug, rot_l10_M0_inv = self.tta_rotation_preprocess(img_np0[:, ::-1], -15)
                    rot_l10_M1, img_ts1_aug, rot_l10_M1_inv = self.tta_rotation_preprocess(img_np1[:, ::-1], -15)
                elif tta_elem == 'eqhist':
                    img_ts0_aug = self.frame2tensor(cv2.equalizeHist(img_np0))
                    img_ts1_aug = self.frame2tensor(cv2.equalizeHist(img_np1))
                elif tta_elem == 'clahe':
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    img_ts0_aug = self.frame2tensor(clahe.apply(img_np0))
                    img_ts1_aug = self.frame2tensor(clahe.apply(img_np1))
                else:
                    raise ValueError('Unknown TTA method.')

                images0.append(img_ts0_aug)
                images1.append(img_ts1_aug)

            # Inference
            if forward_type == 'cross':
                pred = self._superglue_matcher.forward_cross(
                    data={
                        "image0": torch.cat(images0),
                        "image1": torch.cat(images1)
                    },
                    ttas=tta, tta_groups=tta_groups)
            elif forward_type == 'flat':
                pred = self._superglue_matcher.forward_flat(
                data={
                    "image0": torch.cat(images0),
                    "image1": torch.cat(images1)
                },
                ttas=tta, tta_groups=tta_groups)
            else:
                raise RuntimeError(f'Unknown forward_type {forward_type}')

            mkpts0, mkpts1, mconf = [], [], []
            for group_pred in pred:
                pred_aug = {k: v[0].detach().cpu().numpy().squeeze() for k, v in group_pred.items()}
                kpts0, kpts1 = pred_aug["keypoints0"], pred_aug["keypoints1"]
                matches, conf = pred_aug["matches0"], pred_aug["matching_scores0"]

                if self.conf_thresh is None:
                    valid = matches > -1
                else:
                    valid = (matches > -1) & (conf >= self.conf_thresh)
                mkpts0.append(kpts0[valid])
                mkpts1.append(kpts1[matches[valid]])
                mconf.append(conf[valid])

            cat_mkpts0 = np.concatenate(mkpts0)
            cat_mkpts1 = np.concatenate(mkpts1)
            mask0 = (cat_mkpts0[:, 0] >= 0) & (cat_mkpts0[:, 0] < img_np0.shape[1]) & (cat_mkpts0[:, 1] >= 0) & (cat_mkpts0[:, 1] < img_np0.shape[0])
            mask1 = (cat_mkpts1[:, 0] >= 0) & (cat_mkpts1[:, 0] < img_np1.shape[1]) & (cat_mkpts1[:, 1] >= 0) & (cat_mkpts1[:, 1] < img_np1.shape[0])
            return cat_mkpts0[mask0 & mask1] / scale0, cat_mkpts1[mask0 & mask1] / scale1
        

class LoFTRMatcher:
    def __init__(self, device=None, input_longside=1200, conf_th=None):
        self._loftr_matcher = KF.LoFTR(pretrained=None)
        self._loftr_matcher.load_state_dict(torch.load("kornia/loftr_outdoor.ckpt")['state_dict'])
        self._loftr_matcher = self._loftr_matcher.to(device).eval()
        self.device = device
        self.conf_thresh = conf_th
        
    def prep_img(self, img, long_side=1200):
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1]) 
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        else:
            scale = 1.0

        img_ts = K.image_to_tensor(img, False).float() / 255.
        img_ts = K.color.bgr_to_rgb(img_ts)
        img_ts = K.color.rgb_to_grayscale(img_ts)
        return img, img_ts.to(self.device), scale
    
    def tta_rotation_preprocess(self, img_np, angle):
        rot_M = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1)
        rot_M_inv = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1)
        rot_img = cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0]))

        rot_img_ts = K.image_to_tensor(rot_img, False).float() / 255.
        rot_img_ts = K.color.bgr_to_rgb(rot_img_ts)
        rot_img_ts = K.color.rgb_to_grayscale(rot_img_ts)
        return rot_M, rot_img_ts.to(self.device), rot_M_inv

    def tta_rotation_postprocess(self, kpts, img_np, rot_M_inv):
        ones = np.ones(shape=(kpts.shape[0], ), dtype=np.float32)[:, None]
        hom = np.concatenate([kpts, ones], 1)
        rot_kpts = rot_M_inv.dot(hom.T).T[:, :2]
        mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < img_np.shape[1]) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < img_np.shape[0])
        return rot_kpts, mask
# 
    def __call__(self, img_np1, img_np2, input_longside, tta=['orig', 'flip_lr']):
        with torch.no_grad():
            img_np1, img_ts0, scale0 = self.prep_img(img_np1, input_longside)
            img_np2, img_ts1, scale1 = self.prep_img(img_np2, input_longside)
            images0, images1 = [], []

            # TTA
            for tta_elem in tta:
                if tta_elem == 'orig':
                    img_ts0_aug, img_ts1_aug = img_ts0, img_ts1
                elif tta_elem == 'flip_lr':
                    img_ts0_aug = torch.flip(img_ts0, [3, ])
                    img_ts1_aug = torch.flip(img_ts1, [3, ])
                elif tta_elem == 'flip_ud':
                    img_ts0_aug = torch.flip(img_ts0, [2, ])
                    img_ts1_aug = torch.flip(img_ts1, [2, ])
                elif tta_elem == 'rot_r10':
                    rot_r10_M0, img_ts0_aug, rot_r10_M0_inv = self.tta_rotation_preprocess(img_np1, 10)
                    rot_r10_M1, img_ts1_aug, rot_r10_M1_inv = self.tta_rotation_preprocess(img_np2, 10)
                elif tta_elem == 'rot_l10':
                    rot_l10_M0, img_ts0_aug, rot_l10_M0_inv = self.tta_rotation_preprocess(img_np1, -10)
                    rot_l10_M1, img_ts1_aug, rot_l10_M1_inv = self.tta_rotation_preprocess(img_np2, -10)
                else:
                    raise ValueError('Unknown TTA method.')
                images0.append(img_ts0_aug)
                images1.append(img_ts1_aug)

            # Inference
            input_dict = {"image0": torch.cat(images0), "image1": torch.cat(images1)}
            correspondences = self._loftr_matcher(input_dict)
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            batch_id = correspondences['batch_indexes'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()

            # Reverse TTA
            for idx, tta_elem in enumerate(tta):
                batch_mask = batch_id == idx

                if tta_elem == 'orig':
                    pass
                elif tta_elem == 'flip_lr':
                    mkpts0[batch_mask, 0] = img_np1.shape[1] - mkpts0[batch_mask, 0]
                    mkpts1[batch_mask, 0] = img_np2.shape[1] - mkpts1[batch_mask, 0]
                elif tta_elem == 'flip_ud':
                    mkpts0[batch_mask, 1] = img_np1.shape[0] - mkpts0[batch_mask, 1]
                    mkpts1[batch_mask, 1] = img_np2.shape[0] - mkpts1[batch_mask, 1]
                elif tta_elem == 'rot_r10':
                    mkpts0[batch_mask], mask0 = self.tta_rotation_postprocess(mkpts0[batch_mask], img_np1, rot_r10_M0_inv)
                    mkpts1[batch_mask], mask1 = self.tta_rotation_postprocess(mkpts1[batch_mask], img_np2, rot_r10_M1_inv)
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(np.float32) * -10.
                elif tta_elem == 'rot_l10':
                    mkpts0[batch_mask], mask0 = self.tta_rotation_postprocess(mkpts0[batch_mask], img_np1, rot_l10_M0_inv)
                    mkpts1[batch_mask], mask1 = self.tta_rotation_postprocess(mkpts1[batch_mask], img_np2, rot_l10_M1_inv)
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(np.float32) * -10.
                else:
                    raise ValueError('Unknown TTA method.')
                    
            if self.conf_thresh is not None:
                th_mask = confidence >= self.conf_thresh
            else:
                th_mask = confidence >= 0.
            mkpts0, mkpts1 = mkpts0[th_mask, :], mkpts1[th_mask, :]

            # Matching points
            return mkpts0 / scale0, mkpts1 / scale1
        

base_config = {
    "superpoint": {
        "nms_radius": 3,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048,
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }
}
f8000_config = {
    "superpoint": {
        "nms_radius": 3,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048*4,
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    }
}
superglue_matcher = SuperGlueMatcherV2(base_config, device=device, conf_th=0.2)     
superglue_matcher_8096 = SuperGlueMatcherV2(f8000_config, device=device, conf_th=0.2)     
loftr_matcher = LoFTRMatcher(device=device, conf_th=0.3)


def get_img_pairs_exhaustive(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i+1, len(img_fnames)):
            index_pairs.append((i,j))
    return index_pairs


def get_image_pairs_shortlist(fnames,
                              sim_th = 0.6, # should be strict
                              min_pairs = 20,
                              exhaustive_if_less = 20,
                              device=torch.device('cpu')):
    num_imgs = len(fnames)

    if num_imgs <= exhaustive_if_less:
        return get_img_pairs_exhaustive(fnames)
    
    model_path = ['./efficientnet/tf_efficientnet_b7.pth']
    model_name = ['tf_efficientnet_b7']
    descs_list = []
    for i in range(len(model_name)): 
        model = timm.create_model(model_name[i], 
                                  checkpoint_path=model_path[i])
        model.eval()
        descs = get_global_desc(fnames, model, device=device)
        descs_list.append(descs)
        
    descs = torch.cat(descs_list, dim=-1)
    print(descs.shape)
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # removing half
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)
    for st_idx in range(num_imgs-1):
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]  
        for idx in to_match:
            if st_idx == idx:
                continue
            if dm[st_idx, idx] < 1000:
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total+=1
    matching_list = sorted(list(set(matching_list)))
    return matching_list


def get_global_desc(fnames, model, device=torch.device("cpu")):
    model = model.eval()
    model = model.to(device)

    config   = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    global_descs_convnext = []

    for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        key = os.path.splitext(os.path.basename(img_fname_full))[0]

        # ──── DEBUG 1: does the file exist? ────
        if not os.path.isfile(img_fname_full):
            print(f"[DEBUG] {key}: file does NOT exist → {img_fname_full}")
            continue

        img = cv2.imread(img_fname_full)

        # ──── DEBUG 2: did imread succeed? ────
        if img is None:
            print(f"[DEBUG] {key}: cv2.imread() returned None")
            continue
        else:
            print(f"[DEBUG] {key}: loaded with shape {img.shape}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))

        # ──── DEBUG 3: post-processing shape check ────
        print(f"[DEBUG] {key}: after resize → {img.shape}")

        img = Image.fromarray(img)
        timg = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            desc     = model.forward_features(timg).mean(dim=(-1, 2))
            desc     = desc.view(1, -1)
            desc_lr  = model.forward_features(timg.flip(-1)).mean(dim=(-1, 2))
            desc_lr  = desc_lr.view(1, -1)
            desc_norm = F.normalize((desc + desc_lr) / 2, dim=1, p=2)

        global_descs_convnext.append(desc_norm.detach().cpu())

    global_descs_all = torch.cat(global_descs_convnext, dim=0)
    return global_descs_all

def match_loftr_superglue(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout_loftr',
                   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                   min_matches=15, resize_to_ = (640, 480)):

    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]

            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)
            
            mkpts0_loftr, mkpts1_loftr = loftr_matcher(img1, img2, 1024)
            mkpts1_loftr_lr, mkpts0_loftr_lr = loftr_matcher(img2, img1, 1024)
            mkpts0_superglue_1024, mkpts1_superglue_1024 = superglue_matcher(img1, img2, 1024, tta_groups=[('orig', 'orig'),('flip_lr', 'flip_lr')])
            mkpts0_superglue_1440, mkpts1_superglue_1440 = superglue_matcher(img1, img2, 1440, tta_groups=[('orig', 'orig'),('flip_lr', 'flip_lr')])
            
            mkpts0 = np.concatenate([mkpts0_loftr,mkpts0_superglue_1024,mkpts0_superglue_1440,mkpts0_loftr_lr], axis=0)
            mkpts1 = np.concatenate([mkpts1_loftr,mkpts1_superglue_1024,mkpts1_superglue_1440,mkpts1_loftr_lr], axis=0)
            
            n_matches = len(mkpts1)
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
#     clean_index_pairs = [(i,j) for pi, (i,j) in enumerate(index_pairs) if pi not in drop_pair]
    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                   ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return

def match_superglue(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout_loftr',
                   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                   min_matches=15, resize_to_ = (640, 480)):

    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for pair_idx in progress_bar(index_pairs):
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]

            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)
            
            mkpts0_superglue_1024, mkpts1_superglue_1024 = superglue_matcher_8096(img1, img2, 1024)
            mkpts0_superglue_1440, mkpts1_superglue_1440 = superglue_matcher_8096(img1, img2, 1440)
            
            mkpts0 = np.concatenate([mkpts0_superglue_1024,mkpts0_superglue_1440], axis=0)
            mkpts1 = np.concatenate([mkpts1_superglue_1024,mkpts1_superglue_1440], axis=0)
            
            n_matches = len(mkpts1)
            group  = f_match.require_group(key1)
            if n_matches >= min_matches:
                 group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
#     clean_index_pairs = [(i,j) for pi, (i,j) in enumerate(index_pairs) if pi not in drop_pair]
    # Let's find unique loftr pixels and group them together.
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts=defaultdict(int)
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f_match:
        for k1 in f_match.keys():
            group  = f_match[k1]
            for k2 in group.keys():
                matches = group[k2][...]
                total_kpts[k1]
                kpts[k1].append(matches[:, :2])
                kpts[k2].append(matches[:, 2:])
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0]+=total_kpts[k1]
                current_match[:, 1]+=total_kpts[k2]
                total_kpts[k1]+=len(matches)
                total_kpts[k2]+=len(matches)
                match_indexes[k1][k2]=current_match

    for k in kpts.keys():
        kpts[k] = np.round(np.concatenate(kpts[k], axis=0))
    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)
    for k in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]),dim=0, return_inverse=True)
        unique_match_idxs[k] = uniq_reverse_idxs
        unique_kpts[k] = uniq_kps.numpy()
    for k1, group in match_indexes.items():
        for k2, m in group.items():
            m2 = deepcopy(m)
            m2[:,0] = unique_match_idxs[k1][m2[:,0]]
            m2[:,1] = unique_match_idxs[k2][m2[:,1]]
            mkpts = np.concatenate([unique_kpts[k1][ m2[:,0]],
                                    unique_kpts[k2][  m2[:,1]],
                                   ],
                                   axis=1)
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[k1][k2] = m2_semiclean2.numpy()
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp:
        for k, kpts1 in unique_kpts.items():
            f_kp[k] = kpts1
    
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        for k1, gr in out_match.items():
            group  = f_match.require_group(k1)
            for k2, match in gr.items():
                group[k2] = match
    return


img_dir = 'examples/images'
img_fnames = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if not filename.startswith('.')]

feature_dir = './featureout'
if not os.path.isdir(feature_dir):
    os.makedirs(feature_dir, exist_ok=True)

index_pairs = get_image_pairs_shortlist(img_fnames,
                        sim_th = 0.5, # should be strict
                        min_pairs = 35, # we select at least min_pairs PER IMAGE with biggest similarity
                        exhaustive_if_less = 20,
                        device=device)

print(index_pairs)


if len(index_pairs) >= 400:
    match_superglue(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(600, 800))
    mapper_options = pycolmap.IncrementalMapperOptions()
    # Use default mapper options - specific IMC options not available in this pycolmap version
else:
    match_loftr_superglue(img_fnames, index_pairs, feature_dir=feature_dir, device=device, resize_to_=(600, 800))
    mapper_options = pycolmap.IncrementalMapperOptions()
    # Use default mapper options - specific IMC options not available in this pycolmap version