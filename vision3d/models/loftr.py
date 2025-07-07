"""
LoFTR (Local Feature TRansformer) matcher implementation.

This module provides a wrapper around the LoFTR dense matching network,
which uses transformer attention to find pixel-level correspondences between images.
"""

import os
import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from typing import Dict, List, Optional, Tuple
import logging

from .base import BaseMatcher

logger = logging.getLogger(__name__)


class LoFTRMatcher(BaseMatcher):
    """
    LoFTR (Local Feature TRansformer) matcher wrapper.
    
    LoFTR is particularly effective for:
    - Textureless regions (walls, sky)
    - Repetitive patterns
    - Large viewpoint changes
    - Challenging lighting conditions
    
    Args:
        device: Torch device for computation
        input_longside: Default long side for image preprocessing
        confidence_threshold: Minimum confidence for valid matches
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        super().__init__(config, device)
        
        # LoFTR uses 10 degree rotations (different from SuperGlue)
        self.ROTATION_ANGLE = 10.0
        
        # Initialize LoFTR model
        self._matcher = KF.LoFTR(pretrained=None)
        
        # Load pretrained weights
        model_path = self.config['model_path']
        if os.path.exists(model_path):
            self._matcher.load_state_dict(torch.load(model_path)['state_dict'])
            logger.info(f"Loaded LoFTR weights from {model_path}")
        else:
            logger.warning(f"LoFTR weights not found at {model_path}, using random initialization")
        
        self._matcher = self._matcher.to(self.device).eval()
    
    def get_default_config(self) -> Dict:
        """Get default LoFTR configuration."""
        return {
            "model_path": "kornia/loftr_outdoor.ckpt",
            "confidence_threshold": 0.3,
            "input_longside": 1200,
            "min_matches": 15,
        }
    
    def preprocess_image(self, img: np.ndarray, long_side: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for LoFTR matching.
        
        Args:
            img: Input image
            long_side: Target long side dimension
            
        Returns:
            Resized image and scale factor
        """
        long_side = long_side or self.config['input_longside']
        
        if long_side is not None:
            scale = long_side / max(img.shape[0], img.shape[1])
            w = int(img.shape[1] * scale)
            h = int(img.shape[0] * scale)
            img_resized = cv2.resize(img, (w, h))
        else:
            scale = 1.0
            img_resized = img
        
        return img_resized, scale
    
    def _image_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert image to LoFTR-compatible tensor."""
        img_tensor = K.image_to_tensor(img, False).float() / 255.0
        img_tensor = K.color.bgr_to_rgb(img_tensor)
        img_tensor = K.color.rgb_to_grayscale(img_tensor)
        return img_tensor.to(self.device)
    
    def tta_rotation_preprocess(self, img_np: np.ndarray, angle: float) -> Tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Apply rotation preprocessing for TTA."""
        rot_M = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), angle, 1)
        rot_M_inv = cv2.getRotationMatrix2D((img_np.shape[1] / 2, img_np.shape[0] / 2), -angle, 1)
        rot_img = cv2.warpAffine(img_np, rot_M, (img_np.shape[1], img_np.shape[0]))
        
        rot_img_ts = self._image_to_tensor(rot_img)
        return rot_M, rot_img_ts, rot_M_inv
    
    def tta_rotation_postprocess(self, kpts: np.ndarray, img_np: np.ndarray, rot_M_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply inverse rotation to keypoints."""
        ones = np.ones(shape=(kpts.shape[0], ), dtype=np.float32)[:, None]
        hom = np.concatenate([kpts, ones], 1)
        rot_kpts = rot_M_inv.dot(hom.T).T[:, :2]
        mask = (rot_kpts[:, 0] >= 0) & (rot_kpts[:, 0] < img_np.shape[1]) & (rot_kpts[:, 1] >= 0) & (rot_kpts[:, 1] < img_np.shape[0])
        return rot_kpts, mask
    
    def match_pair(self, img1_path: str, img2_path: str,
                  input_longside: Optional[int] = None,
                  tta_methods: List[str] = None,
                  confidence_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match keypoints between two images using LoFTR.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            input_longside: Target size for preprocessing
            tta_methods: List of TTA methods to apply
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
        
        tta_methods = tta_methods or ['orig', 'flip_lr']
        confidence_threshold = confidence_threshold or self.config['confidence_threshold']
        
        with torch.no_grad():
            # Preprocess images
            img0_resized, scale0 = self.preprocess_image(img0, input_longside)
            img1_resized, scale1 = self.preprocess_image(img1, input_longside)
            
            # Convert to tensors
            img0_tensor = self._image_to_tensor(img0_resized)
            img1_tensor = self._image_to_tensor(img1_resized)
            
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
            valid_mask = confidence >= confidence_threshold
            
            mkpts0 = mkpts0[valid_mask]
            mkpts1 = mkpts1[valid_mask]
            confidence = confidence[valid_mask]
            
            return mkpts0 / scale0, mkpts1 / scale1, confidence
    
    def match_multi_scale(self, img1_path: str, img2_path: str,
                         scales: List[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match at multiple scales for better coverage.
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            scales: List of scales to use
            
        Returns:
            Combined matches from all scales
        """
        scales = scales or [640, 1024, 1440]
        
        all_mkpts0, all_mkpts1, all_confidence = [], [], []
        
        for scale in scales:
            mkpts0, mkpts1, confidence = self.match_pair(
                img1_path, img2_path, 
                input_longside=scale
            )
            
            if len(mkpts0) > 0:
                all_mkpts0.append(mkpts0)
                all_mkpts1.append(mkpts1)
                all_confidence.append(confidence)
        
        if all_mkpts0:
            # Concatenate all matches
            mkpts0 = np.concatenate(all_mkpts0)
            mkpts1 = np.concatenate(all_mkpts1)
            confidence = np.concatenate(all_confidence)
            
            # Remove duplicate matches (matches that are very close)
            unique_mask = self._get_unique_matches(mkpts0, mkpts1, threshold=3.0)
            
            return mkpts0[unique_mask], mkpts1[unique_mask], confidence[unique_mask]
        
        return np.array([]), np.array([]), np.array([])
    
    def _get_unique_matches(self, mkpts0: np.ndarray, mkpts1: np.ndarray, 
                           threshold: float = 3.0) -> np.ndarray:
        """
        Remove duplicate matches that are within threshold distance.
        
        Args:
            mkpts0, mkpts1: Matched keypoints
            threshold: Distance threshold for uniqueness
            
        Returns:
            Boolean mask of unique matches
        """
        if len(mkpts0) == 0:
            return np.array([], dtype=bool)
        
        # Concatenate keypoints for uniqueness check
        combined = np.concatenate([mkpts0, mkpts1], axis=1)
        
        # Keep track of unique matches
        unique_mask = np.ones(len(combined), dtype=bool)
        
        for i in range(len(combined)):
            if not unique_mask[i]:
                continue
            
            # Check distances to all subsequent points
            distances = np.linalg.norm(combined[i+1:] - combined[i], axis=1)
            duplicates = distances < threshold
            
            # Mark duplicates
            unique_mask[i+1:][duplicates] = False
        
        return unique_mask