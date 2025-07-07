"""
SuperGlue matcher implementation with Test-Time Augmentation support.

This module provides a wrapper around the SuperGlue neural network matcher,
adding support for Test-Time Augmentation (TTA) to improve matching robustness.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import sys
from copy import deepcopy

from .base import BaseMatcher

# Import SuperGlue components
try:
    from superglue.models.superglue import SuperGlue
    from superglue.models.superpoint import SuperPoint
except ImportError:
    print("Error: SuperGlue not found. Please install it from: https://github.com/magicleap/SuperGluePretrainedNetwork")
    sys.exit(1)

logger = logging.getLogger(__name__)


class SuperGlueCustomMatchingV2(torch.nn.Module):
    """
    Enhanced SuperGlue matcher with Test-Time Augmentation support.
    
    This class extends the standard SuperGlue matching pipeline with various
    augmentation techniques to improve matching robustness.
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TTA transformation mappings
        self.tta_map = {
            'orig': self._untta_identity,
            'eqhist': self._untta_identity,
            'clahe': self._untta_identity,
            'flip_lr': self._untta_flip_horizontal,
            'flip_ud': self._untta_flip_vertical,
            'rot_r10': self._untta_rotate_right10,  # Actually rotates 15 degrees
            'rot_l10': self._untta_rotate_left10,    # Actually rotates 15 degrees
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


class SuperGlueMatcher(BaseMatcher):
    """
    High-level wrapper for SuperGlue matching with preprocessing and TTA support.
    
    This class provides an easy-to-use interface for SuperGlue matching,
    handling image preprocessing, Test-Time Augmentation, and post-processing.
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        super().__init__(config, device)
        
        self._matcher = SuperGlueCustomMatchingV2(
            config=self.config, 
            device=self.device
        ).eval().to(self.device)
    
    def get_default_config(self) -> Dict:
        """Get default SuperGlue configuration."""
        return {
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
            "matching": {
                "min_matches": 15,
                "resize_resolution": (640, 480),
                "long_side_resize": 1200,
            }
        }
    
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
            # First flip, then rotate
            flipped = img_gray[:, ::-1]
            rot_matrix = cv2.getRotationMatrix2D(
                (flipped.shape[1] / 2, flipped.shape[0] / 2), angle, 1
            )
            rotated = cv2.warpAffine(flipped, rot_matrix, (flipped.shape[1], flipped.shape[0]))
            return self.to_tensor(rotated)
        else:
            raise ValueError(f"Unknown TTA method: {tta_method}")
    
    def match_pair(self, img1_path: str, img2_path: str,
                  input_longside: Optional[int] = None,
                  tta_groups: List[Tuple[str, str]] = None,
                  confidence_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match keypoints between two images.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            input_longside: Target size for preprocessing
            tta_groups: TTA method groups for matching
            confidence_threshold: Override confidence threshold
            
        Returns:
            Matched keypoints for both images and confidence scores
        """
        # Load images
        img0 = cv2.imread(img1_path)
        img1 = cv2.imread(img2_path)
        
        if img0 is None or img1 is None:
            logger.error(f"Failed to load images: {img1_path} or {img2_path}")
            return np.array([]), np.array([]), np.array([])
        
        tta_groups = tta_groups or [('orig', 'orig')]
        confidence_threshold = confidence_threshold or self.config['superglue']['match_threshold']
        
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
            all_mkpts0, all_mkpts1, all_confidence = [], [], []
            
            for pred in predictions:
                pred_np = {k: v[0].detach().cpu().numpy().squeeze() for k, v in pred.items()}
                kpts0 = pred_np["keypoints0"]
                kpts1 = pred_np["keypoints1"]
                matches = pred_np["matches0"]
                confidence = pred_np["matching_scores0"]
                
                # Filter by confidence
                valid = (matches > -1) & (confidence >= confidence_threshold)
                
                all_mkpts0.append(kpts0[valid])
                all_mkpts1.append(kpts1[matches[valid]])
                all_confidence.append(confidence[valid])
            
            # Concatenate and filter
            mkpts0 = np.concatenate(all_mkpts0) if all_mkpts0 else np.array([])
            mkpts1 = np.concatenate(all_mkpts1) if all_mkpts1 else np.array([])
            confidence = np.concatenate(all_confidence) if all_confidence else np.array([])
            
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
                
                return mkpts0[mask] / scale0, mkpts1[mask] / scale1, confidence[mask]
            
            return np.array([]), np.array([]), np.array([])
    
    def match_with_high_resolution(self, img1_path: str, img2_path: str,
                                  max_keypoints: int = 8192) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match images with higher keypoint limits for better coverage.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            max_keypoints: Maximum number of keypoints to extract
            
        Returns:
            Matched keypoints and confidence scores
        """
        # Create high-resolution config
        high_res_config = deepcopy(self.config)
        high_res_config['superpoint']['max_keypoints'] = max_keypoints
        
        # Temporarily update matcher config
        original_config = self._matcher.superpoint.config
        self._matcher.superpoint.config = high_res_config['superpoint']
        
        # Match with high resolution
        result = self.match_pair(img1_path, img2_path, input_longside=1440)
        
        # Restore original config
        self._matcher.superpoint.config = original_config
        
        return result