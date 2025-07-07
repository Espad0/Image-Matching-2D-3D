#!/usr/bin/env python3
"""
Image Matching Pipeline for 3D Reconstruction

This module provides a comprehensive image matching pipeline using state-of-the-art
deep learning models (SuperGlue and LoFTR) for feature matching in image pairs.
It's designed for 3D reconstruction workflows and includes various augmentation
techniques to improve matching robustness.

Features:
- SuperGlue matcher with Test-Time Augmentation (TTA)
- LoFTR matcher with TTA support
- Efficient image pair selection using global descriptors
- Batch processing with progress tracking
- COLMAP-compatible output format

Usage:
    python image_matcher.py --input_dir ./images --output_dir ./features --method loftr_superglue

Author: Your Name
License: MIT
"""

import argparse
import gc
import logging
import os
import sqlite3
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import h5py
import kornia as K
import kornia.feature as KF
import numpy as np
import pycolmap
import timm
import torch
import torch.nn.functional as F
from fastprogress import progress_bar
from PIL import Image, ExifTags
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# Import SuperGlue components
try:
    from superglue.models.superglue import SuperGlue
    from superglue.models.superpoint import SuperPoint
except ImportError:
    print("Error: SuperGlue not found. Please install it from: https://github.com/magicleap/SuperGluePretrainedNetwork")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "superpoint": {
        "nms_radius": 3,
        "keypoint_threshold": 0.005,
        "max_keypoints": 2048,
    },
    "superglue": {
        "weights": "outdoor",
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
    },
    "loftr": {
        "model_path": "kornia/loftr_outdoor.ckpt",
        "confidence_threshold": 0.3,
    },
    "matching": {
        "min_matches": 15,
        "resize_resolution": (640, 480),
        "long_side_resize": 1200,
    },
    "image_selection": {
        "similarity_threshold": 0.5,
        "min_pairs_per_image": 35,
        "exhaustive_if_less_than": 20,
        "model_name": "tf_efficientnet_b7",
        "model_path": "./efficientnet/tf_efficientnet_b7.pth",
    },
    "colmap": {
        "camera_model": "simple-radial",
        "single_camera": False,
        "max_image_id": 2**31 - 1,
        "focal_prior": 1.2,
    },
    "reconstruction": {
        "min_num_matches": 15,
        "init_min_num_inliers": 100,
        "init_max_error": 4.0,
        "min_model_size": 3,
    }
}


def get_device() -> torch.device:
    """Get the appropriate torch device (CUDA if available, else CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device


class SuperGlueCustomMatchingV2(torch.nn.Module):
    """
    Enhanced SuperGlue matcher with Test-Time Augmentation support.
    
    This class extends the standard SuperGlue matching pipeline with various
    augmentation techniques to improve matching robustness.
    
    Args:
        config: Configuration dictionary for SuperPoint and SuperGlue
        device: Torch device for computation
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        super().__init__()
        config = config or DEFAULT_CONFIG
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))
        self.device = device or get_device()
        
        # TTA transformation mappings
        self.tta_map = {
            'orig': self._untta_identity,
            'eqhist': self._untta_identity,
            'clahe': self._untta_identity,
            'flip_lr': self._untta_flip_horizontal,
            'flip_ud': self._untta_flip_vertical,
            'rot_r10': self._untta_rotate_right10,  # Actually rotates 15 degrees like kaggle-py.py
            'rot_l10': self._untta_rotate_left10,    # Actually rotates 15 degrees like kaggle-py.py
            'fliplr_rotr10': self._untta_fliplr_rotate_right10,
            'fliplr_rotl10': self._untta_fliplr_rotate_left10
        }
        
        # Rotation angles for augmentation
        self.ROTATION_ANGLE = 15.0

    def forward_flat(self, data: Dict[str, torch.Tensor], 
                    ttas: List[str] = None, 
                    tta_groups: List[List[str]] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Run SuperPoint and SuperGlue with flat TTA grouping.
        
        Args:
            data: Dictionary with 'image0' and 'image1' tensors
            ttas: List of TTA methods to apply
            tta_groups: Grouping of TTA methods for matching
            
        Returns:
            List of prediction dictionaries for each TTA group
        """
        ttas = ttas or ['orig']
        tta_groups = tta_groups or [['orig']]
        
        pred = {}
        
        # Extract SuperPoint features if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred.update({k + '0': v for k, v in pred0.items()})
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred.update({k + '1': v for k, v in pred1.items()})
        
        # Apply reverse TTA transformations
        self._apply_reverse_tta(pred, data, ttas)
        
        # Process each TTA group
        data.update(pred)
        group_predictions = []
        
        for tta_group in tta_groups:
            group_data = self._prepare_group_data(data, ttas, tta_group)
            group_pred = {**group_data, **self.superglue(group_data)}
            group_predictions.append(group_pred)
            
        return group_predictions

    def _apply_reverse_tta(self, pred: Dict, data: Dict, ttas: List[str]) -> None:
        """Apply reverse TTA transformations to keypoints."""
        pred['scores0'] = list(pred['scores0'])
        pred['scores1'] = list(pred['scores1'])
        
        for i, tta in enumerate(ttas):
            for idx in ['0', '1']:
                pred[f'keypoints{idx}'][i], pred[f'descriptors{idx}'][i], pred[f'scores{idx}'][i] = \
                    self.tta_map[tta](
                        pred[f'keypoints{idx}'][i], 
                        pred[f'descriptors{idx}'][i], 
                        pred[f'scores{idx}'][i],
                        w=data[f'image{idx}'].shape[3], 
                        h=data[f'image{idx}'].shape[2],
                        mask_illegal=True
                    )

    def _prepare_group_data(self, data: Dict, ttas: List[str], tta_group: List[str]) -> Dict:
        """Prepare data for a specific TTA group."""
        group_mask = torch.tensor([tta in tta_group for tta in ttas], dtype=torch.bool)
        group_data = {}
        
        # Prepare keypoints, descriptors, and scores
        for key in ['keypoints', 'descriptors', 'scores']:
            for idx in ['0', '1']:
                full_key = f'{key}{idx}'
                group_data[full_key] = [data[full_key][i] for i, tta in enumerate(ttas) if tta in tta_group]
        
        # Prepare images
        for idx in ['0', '1']:
            group_data[f'image{idx}'] = data[f'image{idx}'][group_mask, ...]
        
        # Concatenate data
        for k, v in group_data.items():
            if isinstance(v, (list, tuple)):
                if k.startswith('descriptor'):
                    group_data[k] = torch.cat(v, 1)[None, ...]
                else:
                    group_data[k] = torch.cat(v)[None, ...]
            else:
                group_data[k] = torch.flatten(v, 0, 1)[None, ...]
                
        return group_data

    def _untta_identity(self, keypoints: torch.Tensor, descriptors: torch.Tensor, 
                       scores: torch.Tensor, w: int, h: int, 
                       inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Identity transformation (no change)."""
        if not inplace:
            keypoints = keypoints.clone()
        return keypoints, descriptors, scores
    
    def _untta_flip_horizontal(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                              scores: torch.Tensor, w: int, h: int,
                              inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Reverse horizontal flip transformation."""
        if not inplace:
            keypoints = keypoints.clone()
        keypoints[:, 0] = w - keypoints[:, 0] - 1.0
        return keypoints, descriptors, scores

    def _untta_flip_vertical(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                            scores: torch.Tensor, w: int, h: int,
                            inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Reverse vertical flip transformation."""
        if not inplace:
            keypoints = keypoints.clone()
        keypoints[:, 1] = h - keypoints[:, 1] - 1.0
        return keypoints, descriptors, scores

    def _apply_rotation_transform(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                                scores: torch.Tensor, w: int, h: int, angle: float,
                                mask_illegal: bool = True) -> Tuple:
        """Apply rotation transformation to keypoints."""
        rot_matrix = torch.from_numpy(
            cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        ).to(torch.float32).to(self.device)
        
        ones = torch.ones_like(keypoints[:, 0])
        homogeneous = torch.cat([keypoints, ones[:, None]], 1)
        rotated_kpts = torch.matmul(rot_matrix, homogeneous.T).T[:, :2]
        
        if mask_illegal:
            mask = (
                (rotated_kpts[:, 0] >= 0) & (rotated_kpts[:, 0] < w) &
                (rotated_kpts[:, 1] >= 0) & (rotated_kpts[:, 1] < h)
            )
            return rotated_kpts[mask], descriptors[:, mask], scores[mask]
        
        return rotated_kpts, descriptors, scores

    def _untta_rotate_right10(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                             scores: torch.Tensor, w: int, h: int,
                             inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Reverse right rotation transformation (15 degrees despite the name)."""
        return self._apply_rotation_transform(
            keypoints, descriptors, scores, w, h, -self.ROTATION_ANGLE, mask_illegal
        )

    def _untta_rotate_left10(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                            scores: torch.Tensor, w: int, h: int,
                            inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Reverse left rotation transformation (15 degrees despite the name)."""
        return self._apply_rotation_transform(
            keypoints, descriptors, scores, w, h, self.ROTATION_ANGLE, mask_illegal
        )

    def _untta_fliplr_rotate_right10(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                                   scores: torch.Tensor, w: int, h: int,
                                   inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Reverse flip + right rotation transformation (15 degrees despite the name)."""
        rotated = self._apply_rotation_transform(
            keypoints, descriptors, scores, w, h, -self.ROTATION_ANGLE, False
        )
        rotated[0][:, 0] = w - rotated[0][:, 0] - 1.0
        
        if mask_illegal:
            mask = (
                (rotated[0][:, 0] >= 0) & (rotated[0][:, 0] < w) &
                (rotated[0][:, 1] >= 0) & (rotated[0][:, 1] < h)
            )
            return rotated[0][mask], rotated[1][:, mask], rotated[2][mask]
        
        return rotated

    def _untta_fliplr_rotate_left10(self, keypoints: torch.Tensor, descriptors: torch.Tensor,
                                   scores: torch.Tensor, w: int, h: int,
                                   inplace: bool = True, mask_illegal: bool = True) -> Tuple:
        """Reverse flip + left rotation transformation (15 degrees despite the name)."""
        rotated = self._apply_rotation_transform(
            keypoints, descriptors, scores, w, h, self.ROTATION_ANGLE, False
        )
        rotated[0][:, 0] = w - rotated[0][:, 0] - 1.0
        
        if mask_illegal:
            mask = (
                (rotated[0][:, 0] >= 0) & (rotated[0][:, 0] < w) &
                (rotated[0][:, 1] >= 0) & (rotated[0][:, 1] < h)
            )
            return rotated[0][mask], rotated[1][:, mask], rotated[2][mask]
        
        return rotated


class SuperGlueMatcherV2:
    """
    High-level wrapper for SuperGlue matching with preprocessing and TTA support.
    
    Args:
        config: Configuration dictionary
        device: Torch device for computation
        confidence_threshold: Minimum confidence for valid matches
    """
    
    def __init__(self, config: Optional[Dict] = None, 
                 device: Optional[torch.device] = None,
                 confidence_threshold: Optional[float] = None):
        self.config = config or DEFAULT_CONFIG
        self.device = device or get_device()
        self.confidence_threshold = confidence_threshold or self.config['superglue']['match_threshold']
        
        self._matcher = SuperGlueCustomMatchingV2(
            config=self.config, 
            device=self.device
        ).eval().to(self.device)
    
    def preprocess_image(self, img: np.ndarray, long_side: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for matching.
        
        Args:
            img: Input image as numpy array
            long_side: Target size for the longest dimension
            
        Returns:
            Preprocessed grayscale image and scale factor
        """
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1])
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img = cv2.resize(img, (w, h))
        else:
            scale = 1.0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray, scale
    
    def to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy image to torch tensor."""
        return (torch.from_numpy(frame).float() / 255.0)[None, None].to(self.device)
    
    def apply_tta_augmentation(self, img_gray: np.ndarray, img_tensor: torch.Tensor,
                             tta_method: str) -> torch.Tensor:
        """Apply Test-Time Augmentation to image."""
        if tta_method == 'orig':
            return img_tensor
        elif tta_method == 'flip_lr':
            return torch.flip(img_tensor, [3])
        elif tta_method == 'flip_ud':
            return torch.flip(img_tensor, [2])
        elif tta_method == 'eqhist':
            enhanced = cv2.equalizeHist(img_gray)
            return self.to_tensor(enhanced)
        elif tta_method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_gray)
            return self.to_tensor(enhanced)
        elif tta_method.startswith('rot'):
            angle = 15.0 if 'r10' in tta_method else -15.0
            rot_matrix = cv2.getRotationMatrix2D(
                (img_gray.shape[1] / 2, img_gray.shape[0] / 2), angle, 1
            )
            rotated = cv2.warpAffine(img_gray, rot_matrix, (img_gray.shape[1], img_gray.shape[0]))
            return self.to_tensor(rotated)
        elif tta_method.startswith('fliplr'):
            # Handle fliplr_rotr10 and fliplr_rotl10
            angle = 15.0 if 'rotr10' in tta_method else -15.0
            # First flip, then rotate (as in kaggle-py.py)
            flipped = img_gray[:, ::-1]
            rot_matrix = cv2.getRotationMatrix2D(
                (flipped.shape[1] / 2, flipped.shape[0] / 2), angle, 1
            )
            rotated = cv2.warpAffine(flipped, rot_matrix, (flipped.shape[1], flipped.shape[0]))
            return self.to_tensor(rotated)
        else:
            raise ValueError(f"Unknown TTA method: {tta_method}")
    
    def __call__(self, img0: np.ndarray, img1: np.ndarray,
                 input_longside: Optional[int] = None,
                 tta_groups: List[Tuple[str, str]] = None,
                 forward_type: str = 'cross') -> Tuple[np.ndarray, np.ndarray]:
        """
        Match keypoints between two images.
        
        Args:
            img0, img1: Input images as numpy arrays
            input_longside: Target size for preprocessing
            tta_groups: TTA method groups for matching
            forward_type: Matching strategy ('cross' or 'flat')
            
        Returns:
            Matched keypoints for both images
        """
        tta_groups = tta_groups or [('orig', 'orig')]
        
        with torch.no_grad():
            # Preprocess images
            img0_gray, scale0 = self.preprocess_image(img0, input_longside)
            img1_gray, scale1 = self.preprocess_image(img1, input_longside)
            
            # Extract unique TTA methods
            tta_methods = []
            for group in tta_groups:
                tta_methods.extend(group)
            tta_methods = list(set(tta_methods))
            
            # Apply TTA augmentations
            images0, images1 = [], []
            for tta in tta_methods:
                img0_tensor = self.to_tensor(img0_gray)
                img1_tensor = self.to_tensor(img1_gray)
                
                images0.append(self.apply_tta_augmentation(img0_gray, img0_tensor, tta))
                images1.append(self.apply_tta_augmentation(img1_gray, img1_tensor, tta))
            
            # Run matching
            data = {
                "image0": torch.cat(images0),
                "image1": torch.cat(images1)
            }
            
            predictions = self._matcher.forward_flat(
                data=data,
                ttas=tta_methods,
                tta_groups=tta_groups
            )
            
            # Collect matches from all groups
            all_mkpts0, all_mkpts1 = [], []
            
            for pred in predictions:
                pred_np = {k: v[0].detach().cpu().numpy().squeeze() for k, v in pred.items()}
                kpts0 = pred_np["keypoints0"]
                kpts1 = pred_np["keypoints1"]
                matches = pred_np["matches0"]
                confidence = pred_np["matching_scores0"]
                
                # Filter by confidence
                if self.confidence_threshold is None:
                    valid = matches > -1
                else:
                    valid = (matches > -1) & (confidence >= self.confidence_threshold)
                
                all_mkpts0.append(kpts0[valid])
                all_mkpts1.append(kpts1[matches[valid]])
            
            # Concatenate and filter
            mkpts0 = np.concatenate(all_mkpts0) if all_mkpts0 else np.array([])
            mkpts1 = np.concatenate(all_mkpts1) if all_mkpts1 else np.array([])
            
            if len(mkpts0) > 0:
                # Filter out-of-bounds keypoints
                mask0 = (
                    (mkpts0[:, 0] >= 0) & (mkpts0[:, 0] < img0_gray.shape[1]) &
                    (mkpts0[:, 1] >= 0) & (mkpts0[:, 1] < img0_gray.shape[0])
                )
                mask1 = (
                    (mkpts1[:, 0] >= 0) & (mkpts1[:, 0] < img1_gray.shape[1]) &
                    (mkpts1[:, 1] >= 0) & (mkpts1[:, 1] < img1_gray.shape[0])
                )
                mask = mask0 & mask1
                
                return mkpts0[mask] / scale0, mkpts1[mask] / scale1
            
            return np.array([]), np.array([])


class LoFTRMatcher:
    """
    LoFTR (Local Feature TRansformer) matcher wrapper.
    
    Args:
        device: Torch device for computation
        input_longside: Default long side for image preprocessing
        confidence_threshold: Minimum confidence for valid matches
    """
    
    def __init__(self, device: Optional[torch.device] = None,
                 input_longside: int = 1200,
                 confidence_threshold: Optional[float] = None):
        self.device = device or get_device()
        self.input_longside = input_longside
        self.confidence_threshold = confidence_threshold or DEFAULT_CONFIG['loftr']['confidence_threshold']
        # LoFTR uses 10 degree rotations (different from SuperGlue)
        self.ROTATION_ANGLE = 10.0
        
        # Initialize LoFTR model
        self._matcher = KF.LoFTR(pretrained=None)
        
        # Load pretrained weights
        model_path = DEFAULT_CONFIG['loftr']['model_path']
        if os.path.exists(model_path):
            self._matcher.load_state_dict(torch.load(model_path)['state_dict'])
            logger.info(f"Loaded LoFTR weights from {model_path}")
        else:
            logger.warning(f"LoFTR weights not found at {model_path}, using random initialization")
        
        self._matcher = self._matcher.to(self.device).eval()
    
    def preprocess_image(self, img: np.ndarray, long_side: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor, float]:
        """
        Preprocess image for LoFTR matching.
        
        Returns:
            Resized image, tensor, and scale factor
        """
        long_side = long_side or self.input_longside
        
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1])
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img_resized = cv2.resize(img, (w, h))
        else:
            scale = 1.0
            img_resized = img
        
        # Convert to tensor
        img_tensor = K.image_to_tensor(img_resized, False).float() / 255.0
        img_tensor = K.color.bgr_to_rgb(img_tensor)
        img_tensor = K.color.rgb_to_grayscale(img_tensor)
        
        return img_resized, img_tensor.to(self.device), scale
    
    def tta_rotation_preprocess(self, img_np: np.ndarray, angle: float) -> Tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Apply rotation preprocessing for TTA."""
        rot_M = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1)
        rot_M_inv = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1)
        rot_img = cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0]))
        
        rot_img_ts = K.image_to_tensor(rot_img, False).float() / 255.0
        rot_img_ts = K.color.bgr_to_rgb(rot_img_ts)
        rot_img_ts = K.color.rgb_to_grayscale(rot_img_ts)
        return rot_M, rot_img_ts.to(self.device), rot_M_inv
    
    def tta_rotation_postprocess(self, kpts: np.ndarray, img_np: np.ndarray, rot_M_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse rotation to keypoints."""
        ones = np.ones(shape=(kpts.shape[0], ), dtype=np.float32)[:, None]
        hom = np.concatenate([kpts, ones], 1)
        rot_kpts = rot_M_inv.dot(hom.T).T[:, :2]
        mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < img_np.shape[1]) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < img_np.shape[0])
        return rot_kpts, mask
    
    def __call__(self, img0: np.ndarray, img1: np.ndarray,
                 input_longside: Optional[int] = None,
                 tta_methods: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match keypoints between two images using LoFTR.
        
        Args:
            img0, img1: Input images
            input_longside: Target size for preprocessing
            tta_methods: List of TTA methods to apply
            
        Returns:
            Matched keypoints for both images
        """
        tta_methods = tta_methods or ['orig', 'flip_lr']
        
        with torch.no_grad():
            # Preprocess images
            img0_resized, img0_tensor, scale0 = self.preprocess_image(img0, input_longside)
            img1_resized, img1_tensor, scale1 = self.preprocess_image(img1, input_longside)
            
            images0, images1 = [], []
            rot_matrices = {}  # Store rotation matrices for postprocessing
            
            # Apply TTA
            for tta in tta_methods:
                if tta == 'orig':
                    images0.append(img0_tensor)
                    images1.append(img1_tensor)
                elif tta == 'flip_lr':
                    images0.append(torch.flip(img0_tensor, [3]))
                    images1.append(torch.flip(img1_tensor, [3]))
                elif tta == 'flip_ud':
                    images0.append(torch.flip(img0_tensor, [2]))
                    images1.append(torch.flip(img1_tensor, [2]))
                elif tta == 'rot_r10':
                    rot_M0, img0_rot, rot_M0_inv = self.tta_rotation_preprocess(img0_resized, self.ROTATION_ANGLE)
                    rot_M1, img1_rot, rot_M1_inv = self.tta_rotation_preprocess(img1_resized, self.ROTATION_ANGLE)
                    images0.append(img0_rot)
                    images1.append(img1_rot)
                    rot_matrices[tta] = (rot_M0_inv, rot_M1_inv)
                elif tta == 'rot_l10':
                    rot_M0, img0_rot, rot_M0_inv = self.tta_rotation_preprocess(img0_resized, -self.ROTATION_ANGLE)
                    rot_M1, img1_rot, rot_M1_inv = self.tta_rotation_preprocess(img1_resized, -self.ROTATION_ANGLE)
                    images0.append(img0_rot)
                    images1.append(img1_rot)
                    rot_matrices[tta] = (rot_M0_inv, rot_M1_inv)
                else:
                    logger.warning(f"Unsupported TTA method for LoFTR: {tta}")
            
            # Run inference
            input_dict = {
                "image0": torch.cat(images0),
                "image1": torch.cat(images1)
            }
            
            correspondences = self._matcher(input_dict)
            
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()
            batch_ids = correspondences['batch_indexes'].cpu().numpy()
            
            # Reverse TTA transformations
            for idx, tta in enumerate(tta_methods):
                batch_mask = batch_ids == idx
                
                if tta == 'flip_lr':
                    mkpts0[batch_mask, 0] = img0_resized.shape[1] - mkpts0[batch_mask, 0]
                    mkpts1[batch_mask, 0] = img1_resized.shape[1] - mkpts1[batch_mask, 0]
                elif tta == 'flip_ud':
                    mkpts0[batch_mask, 1] = img0_resized.shape[0] - mkpts0[batch_mask, 1]
                    mkpts1[batch_mask, 1] = img1_resized.shape[0] - mkpts1[batch_mask, 1]
                elif tta in ['rot_r10', 'rot_l10'] and tta in rot_matrices:
                    rot_M0_inv, rot_M1_inv = rot_matrices[tta]
                    mkpts0[batch_mask], mask0 = self.tta_rotation_postprocess(mkpts0[batch_mask], img0_resized, rot_M0_inv)
                    mkpts1[batch_mask], mask1 = self.tta_rotation_postprocess(mkpts1[batch_mask], img1_resized, rot_M1_inv)
                    # Penalize out-of-bounds keypoints
                    confidence[batch_mask] += (~(mask0 & mask1)).astype(np.float32) * -10.
            
            # Filter by confidence
            if self.confidence_threshold is not None:
                valid_mask = confidence >= self.confidence_threshold
            else:
                valid_mask = confidence >= 0.0
            
            mkpts0 = mkpts0[valid_mask]
            mkpts1 = mkpts1[valid_mask]
            
            return mkpts0 / scale0, mkpts1 / scale1


def get_unique_indices(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Get indices of first occurrence of unique elements in tensor."""
    unique, idx, counts = torch.unique(
        tensor, dim=dim, sorted=True, 
        return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices


def get_image_pairs_exhaustive(img_filenames: List[str]) -> List[Tuple[int, int]]:
    """Generate all possible image pairs for exhaustive matching."""
    pairs = []
    n = len(img_filenames)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def compute_global_descriptors(filenames: List[str], 
                             model_name: str = None,
                             model_path: str = None,
                             device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Compute global descriptors for images using a pretrained CNN.
    
    Args:
        filenames: List of image file paths
        model_name: Name of the timm model to use
        model_path: Path to model weights
        device: Torch device
        
    Returns:
        Global descriptors tensor
    """
    device = device or get_device()
    model_name = model_name or DEFAULT_CONFIG['image_selection']['model_name']
    model_path = model_path or DEFAULT_CONFIG['image_selection']['model_path']
    
    # Create model
    if model_path and os.path.exists(model_path):
        model = timm.create_model(model_name, checkpoint_path=model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        model = timm.create_model(model_name, pretrained=True)
        logger.info(f"Using pretrained {model_name}")
    
    model = model.eval().to(device)
    
    # Prepare transforms
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    descriptors = []
    
    for filepath in tqdm(filenames, desc="Computing global descriptors"):
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            continue
        
        try:
            # Load and preprocess image
            img = cv2.imread(filepath)
            if img is None:
                logger.error(f"Failed to load image: {filepath}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            img_pil = Image.fromarray(img)
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                desc = model.forward_features(img_tensor).mean(dim=(-1, 2))
                desc = desc.view(1, -1)
                
                # TTA with horizontal flip
                desc_lr = model.forward_features(img_tensor.flip(-1)).mean(dim=(-1, 2))
                desc_lr = desc_lr.view(1, -1)
                
                # Average and normalize
                desc_avg = F.normalize((desc + desc_lr) / 2, dim=1, p=2)
                descriptors.append(desc_avg.cpu())
                
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
            continue
    
    if descriptors:
        return torch.cat(descriptors, dim=0)
    else:
        return torch.tensor([])


def get_image_pairs_shortlist(filenames: List[str],
                            similarity_threshold: float = None,
                            min_pairs_per_image: int = None,
                            exhaustive_if_less_than: int = None,
                            device: Optional[torch.device] = None) -> List[Tuple[int, int]]:
    """
    Generate image pairs based on global descriptor similarity.
    
    Uses global image descriptors to find similar images and create
    a shortlist of pairs for matching, avoiding exhaustive matching
    for large datasets.
    """
    config = DEFAULT_CONFIG['image_selection']
    similarity_threshold = similarity_threshold or config['similarity_threshold']
    min_pairs_per_image = min_pairs_per_image or config['min_pairs_per_image']
    exhaustive_if_less_than = exhaustive_if_less_than or config['exhaustive_if_less_than']
    
    num_images = len(filenames)
    
    # Use exhaustive matching for small datasets
    if num_images <= exhaustive_if_less_than:
        logger.info(f"Using exhaustive matching for {num_images} images")
        return get_image_pairs_exhaustive(filenames)
    
    logger.info(f"Computing image pairs using global descriptors for {num_images} images")
    
    # Compute global descriptors
    descriptors = compute_global_descriptors(filenames, device=device)
    
    if len(descriptors) == 0:
        logger.error("No descriptors computed, falling back to exhaustive matching")
        return get_image_pairs_exhaustive(filenames)
    
    # Compute pairwise distances
    distances = torch.cdist(descriptors, descriptors, p=2).cpu().numpy()
    
    # Generate pairs based on similarity
    pairs = set()
    indices = np.arange(num_images)
    
    for i in range(num_images - 1):
        # Find similar images
        similar_mask = distances[i] <= similarity_threshold
        similar_indices = indices[similar_mask]
        
        # Ensure minimum pairs per image
        if len(similar_indices) < min_pairs_per_image:
            similar_indices = np.argsort(distances[i])[:min_pairs_per_image]
        
        # Add pairs
        for j in similar_indices:
            if i != j and distances[i, j] < 1000:  # Sanity check
                pairs.add(tuple(sorted((i, int(j)))))
    
    pairs_list = sorted(list(pairs))
    logger.info(f"Generated {len(pairs_list)} image pairs")
    
    return pairs_list


def process_matches_to_unique_keypoints(feature_dir: str) -> None:
    """
    Process raw matches to extract unique keypoints and clean match indices.
    
    This function reads the raw matches, identifies unique keypoints across
    all images, and creates cleaned match indices that reference these
    unique keypoints.
    """
    logger.info("Processing matches to extract unique keypoints")
    
    # Collect all keypoints and match indices
    keypoints = defaultdict(list)
    match_indices = defaultdict(dict)
    total_keypoints = defaultdict(int)
    
    # Read raw matches
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f:
        for key1 in f.keys():
            group = f[key1]
            for key2 in group.keys():
                matches = group[key2][...]
                
                # Extract keypoints
                keypoints[key1].append(matches[:, :2])
                keypoints[key2].append(matches[:, 2:])
                
                # Create match indices
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_keypoints[key1]
                current_match[:, 1] += total_keypoints[key2]
                
                total_keypoints[key1] += len(matches)
                total_keypoints[key2] += len(matches)
                match_indices[key1][key2] = current_match
    
    # Round keypoints for unique detection
    for k in keypoints.keys():
        keypoints[k] = np.round(np.concatenate(keypoints[k], axis=0))
    
    # Find unique keypoints
    unique_keypoints = {}
    unique_match_indices = {}
    
    for k in keypoints.keys():
        unique_kpts, unique_reverse_indices = torch.unique(
            torch.from_numpy(keypoints[k]),
            dim=0,
            return_inverse=True
        )
        unique_match_indices[k] = unique_reverse_indices
        unique_keypoints[k] = unique_kpts.numpy()
    
    # Create cleaned matches
    cleaned_matches = defaultdict(dict)
    
    for k1, group in match_indices.items():
        for k2, matches in group.items():
            # Remap to unique indices
            remapped = deepcopy(matches)
            remapped[:, 0] = unique_match_indices[k1][remapped[:, 0]]
            remapped[:, 1] = unique_match_indices[k2][remapped[:, 1]]
            
            # Remove duplicates
            mkpts = np.concatenate([
                unique_keypoints[k1][remapped[:, 0]],
                unique_keypoints[k2][remapped[:, 1]]
            ], axis=1)
            
            unique_indices = get_unique_indices(torch.from_numpy(mkpts), dim=0)
            remapped = remapped[unique_indices]
            
            # Ensure one-to-one matching
            unique_indices1 = get_unique_indices(remapped[:, 0], dim=0)
            remapped = remapped[unique_indices1]
            
            unique_indices2 = get_unique_indices(remapped[:, 1], dim=0)
            remapped = remapped[unique_indices2]
            
            cleaned_matches[k1][k2] = remapped.numpy()
    
    # Save unique keypoints
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f:
        for k, kpts in unique_keypoints.items():
            f[k] = kpts
    
    # Save cleaned matches
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f:
        for k1, group in cleaned_matches.items():
            group_h5 = f.require_group(k1)
            for k2, matches in group.items():
                group_h5[k2] = matches
    
    logger.info("Finished processing matches")


def match_images_loftr_superglue(img_filenames: List[str],
                                index_pairs: List[Tuple[int, int]],
                                feature_dir: str = './features',
                                device: Optional[torch.device] = None,
                                min_matches: int = 15) -> None:
    """
    Match images using both LoFTR and SuperGlue for robust matching.
    
    This function combines matches from both methods to improve robustness
    and coverage of the matching process.
    """
    device = device or get_device()
    
    # Initialize matchers
    loftr_matcher = LoFTRMatcher(device=device)
    superglue_matcher = SuperGlueMatcherV2(DEFAULT_CONFIG, device=device)
    
    logger.info(f"Matching {len(index_pairs)} image pairs using LoFTR + SuperGlue")
    
    # Create output directory
    os.makedirs(feature_dir, exist_ok=True)
    
    # Process matches
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for idx1, idx2 in progress_bar(index_pairs):
            fname1 = img_filenames[idx1]
            fname2 = img_filenames[idx2]
            key1 = os.path.basename(fname1)
            key2 = os.path.basename(fname2)
            
            try:
                # Load images
                img1 = cv2.imread(fname1)
                img2 = cv2.imread(fname2)
                
                if img1 is None or img2 is None:
                    logger.error(f"Failed to load images: {fname1} or {fname2}")
                    continue
                
                # Get matches from different methods
                mkpts0_loftr, mkpts1_loftr = loftr_matcher(img1, img2, 1024)
                mkpts1_loftr_lr, mkpts0_loftr_lr = loftr_matcher(img2, img1, 1024)
                mkpts0_sg_1024, mkpts1_sg_1024 = superglue_matcher(
                    img1, img2, 1024,
                    tta_groups=[('orig', 'orig'), ('flip_lr', 'flip_lr')]
                )
                mkpts0_sg_1440, mkpts1_sg_1440 = superglue_matcher(
                    img1, img2, 1440,
                    tta_groups=[('orig', 'orig'), ('flip_lr', 'flip_lr')]
                )
                
                # Combine all matches
                mkpts0 = np.concatenate([
                    mkpts0_loftr, mkpts0_sg_1024, mkpts0_sg_1440, mkpts0_loftr_lr
                ], axis=0)
                mkpts1 = np.concatenate([
                    mkpts1_loftr, mkpts1_sg_1024, mkpts1_sg_1440, mkpts1_loftr_lr
                ], axis=0)
                
                # Save if enough matches
                if len(mkpts0) >= min_matches:
                    group = f_match.require_group(key1)
                    group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
                    logger.debug(f"Matched {key1} - {key2}: {len(mkpts0)} matches")
                else:
                    logger.warning(f"Too few matches for {key1} - {key2}: {len(mkpts0)}")
                    
            except Exception as e:
                logger.error(f"Error matching {fname1} - {fname2}: {e}")
                continue
    
    # Process to unique keypoints
    process_matches_to_unique_keypoints(feature_dir)


def match_images_superglue(img_filenames: List[str],
                          index_pairs: List[Tuple[int, int]],
                          feature_dir: str = './features',
                          device: Optional[torch.device] = None,
                          min_matches: int = 15) -> None:
    """
    Match images using only SuperGlue with higher keypoint limits.
    
    This is faster than the combined approach and suitable when
    high-quality features are more important than coverage.
    """
    device = device or get_device()
    
    # Create high-resolution config
    high_res_config = deepcopy(DEFAULT_CONFIG)
    high_res_config['superpoint']['max_keypoints'] = 8192
    
    # Initialize matcher
    superglue_matcher = SuperGlueMatcherV2(high_res_config, device=device)
    
    logger.info(f"Matching {len(index_pairs)} image pairs using SuperGlue")
    
    # Create output directory
    os.makedirs(feature_dir, exist_ok=True)
    
    # Process matches
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='w') as f_match:
        for idx1, idx2 in progress_bar(index_pairs):
            fname1 = img_filenames[idx1]
            fname2 = img_filenames[idx2]
            key1 = os.path.basename(fname1)
            key2 = os.path.basename(fname2)
            
            try:
                # Load images
                img1 = cv2.imread(fname1)
                img2 = cv2.imread(fname2)
                
                if img1 is None or img2 is None:
                    logger.error(f"Failed to load images: {fname1} or {fname2}")
                    continue
                
                # Get matches at multiple scales
                mkpts0_1024, mkpts1_1024 = superglue_matcher(img1, img2, 1024)
                mkpts0_1440, mkpts1_1440 = superglue_matcher(img1, img2, 1440)
                
                # Combine matches
                mkpts0 = np.concatenate([mkpts0_1024, mkpts0_1440], axis=0)
                mkpts1 = np.concatenate([mkpts1_1024, mkpts1_1440], axis=0)
                
                # Save if enough matches
                if len(mkpts0) >= min_matches:
                    group = f_match.require_group(key1)
                    group.create_dataset(key2, data=np.concatenate([mkpts0, mkpts1], axis=1))
                    logger.debug(f"Matched {key1} - {key2}: {len(mkpts0)} matches")
                else:
                    logger.warning(f"Too few matches for {key1} - {key2}: {len(mkpts0)}")
                    
            except Exception as e:
                logger.error(f"Error matching {fname1} - {fname2}: {e}")
                continue
    
    # Process to unique keypoints
    process_matches_to_unique_keypoints(feature_dir)


# COLMAP Database utilities
def array_to_blob(array: np.ndarray) -> bytes:
    """Convert numpy array to blob for SQLite storage."""
    return array.tobytes()


def image_ids_to_pair_id(image_id1: int, image_id2: int, max_image_id: int = None) -> int:
    """Convert pair of image IDs to unique pair ID."""
    max_image_id = max_image_id or DEFAULT_CONFIG['colmap']['max_image_id']
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * max_image_id + image_id2


def pair_id_to_image_ids(pair_id: int, max_image_id: int = None) -> Tuple[int, int]:
    """Convert pair ID back to image IDs."""
    max_image_id = max_image_id or DEFAULT_CONFIG['colmap']['max_image_id']
    image_id2 = pair_id % max_image_id
    image_id1 = (pair_id - image_id2) // max_image_id
    return image_id1, image_id2


class COLMAPDatabase(sqlite3.Connection):
    """
    SQLite database interface for COLMAP.
    
    Provides methods to create and populate a COLMAP database with
    cameras, images, keypoints, and matches.
    """
    
    @staticmethod
    def connect(database_path: str) -> 'COLMAPDatabase':
        """Connect to or create a COLMAP database."""
        return sqlite3.connect(database_path, factory=COLMAPDatabase)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_table_creators()
    
    def _setup_table_creators(self) -> None:
        """Setup table creation methods."""
        max_image_id = DEFAULT_CONFIG['colmap']['max_image_id']
        
        self.create_tables = lambda: self.executescript(self._get_create_all_script(max_image_id))
        self.create_cameras_table = lambda: self.executescript(self._get_cameras_table_sql())
        self.create_descriptors_table = lambda: self.executescript(self._get_descriptors_table_sql())
        self.create_images_table = lambda: self.executescript(self._get_images_table_sql(max_image_id))
        self.create_two_view_geometries_table = lambda: self.executescript(self._get_two_view_geometries_table_sql())
        self.create_keypoints_table = lambda: self.executescript(self._get_keypoints_table_sql())
        self.create_matches_table = lambda: self.executescript(self._get_matches_table_sql())
        self.create_name_index = lambda: self.executescript("CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)")
    
    @staticmethod
    def _get_cameras_table_sql() -> str:
        return """CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB,
            prior_focal_length INTEGER NOT NULL)"""
    
    @staticmethod
    def _get_descriptors_table_sql() -> str:
        return """CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
    
    @staticmethod
    def _get_images_table_sql(max_image_id: int) -> str:
        return f"""CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {max_image_id}),
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))"""
    
    @staticmethod
    def _get_two_view_geometries_table_sql() -> str:
        return """CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB)"""
    
    @staticmethod
    def _get_keypoints_table_sql() -> str:
        return """CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB,
            FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
    
    @staticmethod
    def _get_matches_table_sql() -> str:
        return """CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB)"""
    
    def _get_create_all_script(self, max_image_id: int) -> str:
        """Get SQL script to create all tables."""
        return "; ".join([
            self._get_cameras_table_sql(),
            self._get_images_table_sql(max_image_id),
            self._get_keypoints_table_sql(),
            self._get_descriptors_table_sql(),
            self._get_matches_table_sql(),
            self._get_two_view_geometries_table_sql(),
            "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
        ])
    
    def add_camera(self, model: int, width: int, height: int, 
                   params: np.ndarray, prior_focal_length: bool = False,
                   camera_id: Optional[int] = None) -> int:
        """Add a camera to the database."""
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params), prior_focal_length)
        )
        return cursor.lastrowid
    
    def add_image(self, name: str, camera_id: int,
                  prior_q: np.ndarray = None, prior_t: np.ndarray = None,
                  image_id: Optional[int] = None) -> int:
        """Add an image to the database."""
        prior_q = prior_q if prior_q is not None else np.array([1.0, 0.0, 0.0, 0.0])
        prior_t = prior_t if prior_t is not None else np.zeros(3)
        
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, 
             prior_q[0], prior_q[1], prior_q[2], prior_q[3],
             prior_t[0], prior_t[1], prior_t[2])
        )
        return cursor.lastrowid
    
    def add_keypoints(self, image_id: int, keypoints: np.ndarray) -> None:
        """Add keypoints for an image."""
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]
        
        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),)
        )
    
    def add_descriptors(self, image_id: int, descriptors: np.ndarray) -> None:
        """Add descriptors for an image."""
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),)
        )
    
    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray) -> None:
        """Add matches between two images."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),)
        )
    
    def add_two_view_geometry(self, image_id1: int, image_id2: int,
                            matches: np.ndarray,
                            F: np.ndarray = None,
                            E: np.ndarray = None,
                            H: np.ndarray = None,
                            config: int = 2) -> None:
        """Add two-view geometry for an image pair."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        
        F = F if F is not None else np.eye(3)
        E = E if E is not None else np.eye(3)
        H = H if H is not None else np.eye(3)
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H))
        )


def get_focal_length(image_path: str, err_on_default: bool = False) -> float:
    """
    Extract focal length from image EXIF data.
    
    Args:
        image_path: Path to the image file
        err_on_default: If True, raise error when focal length not found
        
    Returns:
        Focal length in pixels
    """
    image = Image.open(image_path)
    max_size = max(image.size)
    
    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
        
        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")
        
        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
        logger.warning(f"No focal length found for {image_path}, using prior {FOCAL_PRIOR}")
    
    return focal


def create_camera(db: COLMAPDatabase, image_path: str, camera_model: str) -> int:
    """
    Create a camera entry in the database.
    
    Args:
        db: COLMAP database connection
        image_path: Path to the image
        camera_model: Camera model type
        
    Returns:
        Camera ID
    """
    image = Image.open(image_path)
    width, height = image.size
    
    focal = get_focal_length(image_path)
    
    if camera_model == 'simple-pinhole':
        model = 0  # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    elif camera_model == 'pinhole':
        model = 1  # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == 'simple-radial':
        model = 2  # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == 'opencv':
        model = 4  # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
    else:
        raise ValueError(f"Unknown camera model: {camera_model}")
         
    return db.add_camera(model, width, height, param_arr)


def import_features_to_colmap(db: COLMAPDatabase,
                            image_dir: str,
                            feature_dir: str,
                            camera_model: str = 'simple-radial',
                            single_camera: bool = False) -> Dict[str, int]:
    """
    Import keypoints and descriptors to COLMAP database.
    
    Args:
        db: COLMAP database connection
        image_dir: Directory containing images
        feature_dir: Directory containing features
        camera_model: Camera model to use
        single_camera: If True, use same camera for all images
        
    Returns:
        Mapping from filename to image ID
    """
    keypoint_file = h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r')
    
    camera_id = None
    fname_to_id = {}
    
    logger.info("Importing keypoints to COLMAP database")
    
    for filename in tqdm(list(keypoint_file.keys()), desc="Importing features"):
        keypoints = np.array(keypoint_file[filename])
        
        # Check if image exists
        image_path = os.path.join(image_dir, filename)
        if not os.path.isfile(image_path):
            logger.error(f"Image not found: {image_path}")
            continue
        
        # Create camera if needed
        if camera_id is None or not single_camera:
            camera_id = create_camera(db, image_path, camera_model)
        
        # Add image
        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id
        
        # Add keypoints
        db.add_keypoints(image_id, keypoints)
    
    keypoint_file.close()
    return fname_to_id


def import_matches_to_colmap(db: COLMAPDatabase,
                           feature_dir: str,
                           fname_to_id: Dict[str, int]) -> None:
    """
    Import matches to COLMAP database.
    
    Args:
        db: COLMAP database connection
        feature_dir: Directory containing matches
        fname_to_id: Mapping from filename to image ID
    """
    match_file = h5py.File(os.path.join(feature_dir, 'matches.h5'), 'r')
    
    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2
    
    logger.info("Importing matches to COLMAP database")
    
    with tqdm(total=n_total, desc="Importing matches") as pbar:
        for key1 in match_file.keys():
            group = match_file[key1]
            for key2 in group.keys():
                if key1 not in fname_to_id or key2 not in fname_to_id:
                    logger.warning(f"Skipping match {key1} - {key2}: image not in database")
                    continue
                
                id1 = fname_to_id[key1]
                id2 = fname_to_id[key2]
                
                pair_id = image_ids_to_pair_id(id1, id2)
                if pair_id in added:
                    logger.warning(f"Duplicate pair: {key1} - {key2}")
                    continue
                
                matches = np.array(group[key2])
                db.add_matches(id1, id2, matches)
                
                added.add(pair_id)
                pbar.update(1)
    
    match_file.close()


def run_colmap_reconstruction(database_path: str,
                            image_path: str,
                            output_path: str,
                            min_num_matches: int = 15,
                            exhaustive_matching: bool = True) -> Dict[int, Any]:
    """
    Run COLMAP incremental reconstruction.
    
    Args:
        database_path: Path to COLMAP database
        image_path: Path to image directory
        output_path: Path for reconstruction output
        min_num_matches: Minimum number of matches for valid image pairs
        exhaustive_matching: If True, run exhaustive matching verification
        
    Returns:
        Reconstruction results
    """
    logger.info("Running COLMAP reconstruction")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Configure pipeline options - exactly as in kaggle-colmap.py
    pipeline_options = pycolmap.IncrementalPipelineOptions()
    # Configure mapper options through the pipeline options
    mapper_options = pipeline_options.mapper
    # You can now set specific mapper options if needed, for example:
    # mapper_options.init_min_num_inliers = 100
    # mapper_options.init_max_error = 4.0
    
    # Run exhaustive matching if requested
    if exhaustive_matching:
        logger.info("Running exhaustive matching verification")
        pycolmap.match_exhaustive(database_path)
    
    # Run incremental mapping - exactly as in kaggle-colmap.py
    # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=pipeline_options
    )
    
    logger.info(f"Reconstruction complete. Found {len(maps)} models")
    
    # Log reconstruction statistics
    for idx, reconstruction in maps.items():
        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)
        logger.info(f"Model {idx}: {num_images} images, {num_points} 3D points")
    
    return maps


def export_reconstruction(reconstruction: Any, output_path: str, format: str = 'ply') -> None:
    """
    Export reconstruction to various formats.
    
    Args:
        reconstruction: COLMAP reconstruction object
        output_path: Output file path
        format: Export format ('ply', 'nvm', 'bundler', 'vrml')
    """
    try:
        if format == 'ply':
            reconstruction.write_text(output_path)
            logger.info(f"Exported reconstruction to {output_path} in text format")
        else:
            logger.warning(f"Export format {format} not implemented, skipping export")
    except Exception as e:
        logger.error(f"Failed to export reconstruction: {e}")


def main():
    """Main entry point for the image matching pipeline."""
    parser = argparse.ArgumentParser(
        description="Image matching pipeline for 3D reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Match images using LoFTR + SuperGlue (recommended for most cases)
  python image_matcher.py --input_dir ./images --output_dir ./features
  
  # Use SuperGlue only for faster processing
  python image_matcher.py --input_dir ./images --output_dir ./features --method superglue
  
  # Adjust similarity threshold for pair selection
  python image_matcher.py --input_dir ./images --similarity_threshold 0.7
  
  # Use exhaustive matching for small datasets
  python image_matcher.py --input_dir ./images --exhaustive
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='examples/images',
                       help='Directory containing input images (default: examples/images)')
    parser.add_argument('--output_dir', type=str, default='./featureout',
                       help='Directory for output features and matches (default: ./featureout)')
    parser.add_argument('--method', type=str, default='loftr_superglue',
                       choices=['loftr_superglue', 'superglue'],
                       help='Matching method to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for computation')
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                       help='Similarity threshold for image pair selection (default: 0.5)')
    parser.add_argument('--min_pairs', type=int, default=35,
                       help='Minimum number of pairs per image (default: 35)')
    parser.add_argument('--min_matches', type=int, default=15,
                       help='Minimum number of matches to save a pair')
    parser.add_argument('--exhaustive', action='store_true',
                       help='Use exhaustive matching (all pairs)')
    parser.add_argument('--exhaustive_threshold', type=int, default=20,
                       help='Use exhaustive matching if fewer images than this')
    parser.add_argument('--large_dataset_threshold', type=int, default=400,
                       help='Threshold for switching to faster SuperGlue-only matching')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # COLMAP reconstruction options
    parser.add_argument('--skip_reconstruction', action='store_true',
                       help='Skip COLMAP reconstruction after matching')
    parser.add_argument('--camera_model', type=str, default='simple-radial',
                       choices=['simple-pinhole', 'pinhole', 'simple-radial', 'radial', 'opencv'],
                       help='Camera model for COLMAP')
    parser.add_argument('--single_camera', action='store_true',
                       help='Use single camera for all images')
    parser.add_argument('--skip_geometric_verification', action='store_true',
                       help='Skip geometric verification in COLMAP')
    parser.add_argument('--export_format', type=str, default='ply',
                       choices=['ply', 'nvm', 'bundler', 'vrml'],
                       help='Export format for reconstruction')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    img_filenames = []
    
    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.startswith('.') and any(filename.lower().endswith(ext) for ext in image_extensions):
            img_filenames.append(os.path.join(args.input_dir, filename))
    
    if not img_filenames:
        logger.error(f"No images found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(img_filenames)} images")
    
    # Generate image pairs
    if args.exhaustive:
        index_pairs = get_image_pairs_exhaustive(img_filenames)
    else:
        index_pairs = get_image_pairs_shortlist(
            img_filenames,
            similarity_threshold=args.similarity_threshold,
            min_pairs_per_image=args.min_pairs,
            exhaustive_if_less_than=args.exhaustive_threshold,
            device=device
        )
    
    logger.info(f"Generated {len(index_pairs)} image pairs")
    
    # Run matching
    if len(index_pairs) >= args.large_dataset_threshold and args.method == 'loftr_superglue':
        logger.info("Large dataset detected, switching to SuperGlue-only matching for efficiency")
        match_images_superglue(
            img_filenames, index_pairs,
            feature_dir=args.output_dir,
            device=device,
            min_matches=args.min_matches
        )
    else:
        if args.method == 'loftr_superglue':
            match_images_loftr_superglue(
                img_filenames, index_pairs,
                feature_dir=args.output_dir,
                device=device,
                min_matches=args.min_matches
            )
        else:
            match_images_superglue(
                img_filenames, index_pairs,
                feature_dir=args.output_dir,
                device=device,
                min_matches=args.min_matches
            )
    
    # Create COLMAP mapper options (for reference)
    mapper_options = pycolmap.IncrementalMapperOptions()
    logger.info("Matching complete. Ready for COLMAP reconstruction.")
    logger.info(f"Features saved to: {args.output_dir}/")
    
    # Print summary
    try:
        with h5py.File(f'{args.output_dir}/matches.h5', mode='r') as f:
            total_matches = sum(len(group.keys()) for group in f.values())
            logger.info(f"Total image pairs with matches: {total_matches}")
    except:
        pass
    
    # Run COLMAP reconstruction by default (unless skipped)
    if not args.skip_reconstruction:
        logger.info("\nStarting COLMAP reconstruction pipeline")
        
        # Create database
        database_path = os.path.join(args.output_dir, 'colmap.db')
        if os.path.exists(database_path):
            os.remove(database_path)
            logger.info(f"Removed existing database: {database_path}")
        
        # Connect to database
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        
        # Import features
        fname_to_id = import_features_to_colmap(
            db,
            args.input_dir,
            args.output_dir,
            camera_model=args.camera_model,
            single_camera=args.single_camera
        )
        
        # Import matches
        import_matches_to_colmap(db, args.output_dir, fname_to_id)
        
        # Commit database
        db.commit()
        db.close()
        
        # Run reconstruction
        reconstruction_path = os.path.join(args.output_dir, f'colmap_reconstruction')
        maps = run_colmap_reconstruction(
            database_path,
            args.input_dir,
            reconstruction_path,
            min_num_matches=args.min_matches,
            exhaustive_matching=True  # Always run exhaustive matching like kaggle-colmap.py
        )
        
        # Export reconstruction
        if maps:
            for idx, reconstruction in maps.items():
                export_path = os.path.join(
                    reconstruction_path,
                    f'model_{idx}.{args.export_format}'
                )
                export_reconstruction(reconstruction, export_path, args.export_format)
                
                # Save statistics
                stats_path = os.path.join(reconstruction_path, f'model_{idx}_stats.txt')
                with open(stats_path, 'w') as f:
                    f.write(f"Reconstruction Statistics\n")
                    f.write(f"========================\n")
                    f.write(f"Number of images: {len(reconstruction.images)}\n")
                    f.write(f"Number of 3D points: {len(reconstruction.points3D)}\n")
                
                logger.info(f"Exported model {idx} to {export_path}")
        else:
            logger.warning("No successful reconstructions found")
    
    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    main()