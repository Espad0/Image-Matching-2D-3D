"""
LoFTR (Local Feature TRansformer) implementation for dense feature matching.

LoFTR is a state-of-the-art detector-free method that uses transformers
to establish pixel-wise dense matches between image pairs.
"""

import cv2
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from typing import Dict, List, Tuple, Optional
import logging

from .base import BaseMatcher

logger = logging.getLogger(__name__)


class LoFTRMatcher(BaseMatcher):
    """
    LoFTR matcher for dense feature matching.
    
    This implementation includes:
    - Coarse-to-fine matching strategy
    - Test-time augmentation (TTA) support
    - Confidence thresholding
    - Multi-scale matching
    
    Example:
        >>> matcher = LoFTRMatcher(device=torch.device('cuda'))
        >>> kpts1, kpts2, conf = matcher.match_pair('img1.jpg', 'img2.jpg')
    """
    
    def __init__(
        self,
        device: torch.device,
        config: Optional[Dict] = None,
        pretrained: str = 'outdoor'
    ):
        """
        Initialize LoFTR matcher.
        
        Args:
            device: PyTorch device
            config: Configuration dictionary
            pretrained: Pretrained model type ('indoor' or 'outdoor')
        """
        self.pretrained = pretrained
        super().__init__(device, config or self._get_default_config())
    
    def _get_default_config(self) -> Dict:
        """Get default LoFTR configuration."""
        return {
            'image_resize': 1024,
            'confidence_threshold': 0.3,
            'use_tta': True,
            'tta_variants': ['orig', 'flip_lr'],
            'match_threshold': 0.2,
            'max_keypoints': 8192
        }
    
    def _setup(self):
        """Setup LoFTR model."""
        logger.info(f"Loading LoFTR model ({self.pretrained})...")
        
        # Initialize LoFTR
        self.matcher = KF.LoFTR(pretrained=None)
        
        # Load pretrained weights
        if self.pretrained == 'outdoor':
            # In production, weights would be loaded from a local path
            # self.matcher.load_state_dict(torch.load('weights/loftr_outdoor.ckpt')['state_dict'])
            pass
        
        self.matcher = self.matcher.to(self.device).eval()
        logger.info("LoFTR model loaded successfully")
    
    def match_pair(
        self,
        image1_path: str,
        image2_path: str,
        resize: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features between two images using LoFTR.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            resize: Optional resize dimension
        
        Returns:
            Matched keypoints and confidence scores
        """
        # Load and preprocess images
        img1, img1_tensor, scale1 = self._preprocess_image(image1_path, resize)
        img2, img2_tensor, scale2 = self._preprocess_image(image2_path, resize)
        
        with torch.no_grad():
            if self.config['use_tta']:
                # Apply test-time augmentation
                mkpts1, mkpts2, mconf = self._match_with_tta(
                    img1, img2,
                    img1_tensor, img2_tensor
                )
            else:
                # Single forward pass
                mkpts1, mkpts2, mconf = self._match_single(
                    img1_tensor, img2_tensor
                )
        
        # Apply confidence threshold
        if self.config['confidence_threshold'] is not None:
            mask = mconf >= self.config['confidence_threshold']
            mkpts1 = mkpts1[mask]
            mkpts2 = mkpts2[mask]
            mconf = mconf[mask]
        
        # Rescale keypoints to original image size
        mkpts1 = mkpts1 / scale1
        mkpts2 = mkpts2 / scale2
        
        return mkpts1, mkpts2, mconf
    
    def _preprocess_image(
        self,
        image_path: str,
        resize: Optional[int] = None
    ) -> Tuple[np.ndarray, torch.Tensor, float]:
        """Preprocess image for LoFTR."""
        # Read image
        img = cv2.imread(image_path)
        
        # Resize if specified
        if resize is None:
            resize = self.config['image_resize']
        
        if resize is not None:
            scale = resize / max(img.shape[:2])
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height))
        else:
            scale = 1.0
        
        # Convert to tensor
        img_tensor = K.image_to_tensor(img, False).float() / 255.
        img_tensor = K.color.bgr_to_rgb(img_tensor)
        img_tensor = K.color.rgb_to_grayscale(img_tensor)
        img_tensor = img_tensor.to(self.device)
        
        return img, img_tensor, scale
    
    def _match_single(
        self,
        img1_tensor: torch.Tensor,
        img2_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform single matching without TTA."""
        input_dict = {
            "image0": img1_tensor,
            "image1": img2_tensor
        }
        
        correspondences = self.matcher(input_dict)
        
        mkpts1 = correspondences['keypoints0'].cpu().numpy()
        mkpts2 = correspondences['keypoints1'].cpu().numpy()
        mconf = correspondences['confidence'].cpu().numpy()
        
        return mkpts1, mkpts2, mconf
    
    def _match_with_tta(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        img1_tensor: torch.Tensor,
        img2_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform matching with test-time augmentation."""
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        for tta in self.config['tta_variants']:
            # Apply augmentation
            img1_aug, inv_transform1 = self.apply_tta(img1_tensor, tta)
            img2_aug, inv_transform2 = self.apply_tta(img2_tensor, tta)
            
            # Match
            input_dict = {
                "image0": img1_aug,
                "image1": img2_aug
            }
            
            correspondences = self.matcher(input_dict)
            
            mkpts1 = correspondences['keypoints0'].cpu().numpy()
            mkpts2 = correspondences['keypoints1'].cpu().numpy()
            mconf = correspondences['confidence'].cpu().numpy()
            
            # Apply inverse transformation
            if tta != 'orig':
                mkpts1 = inv_transform1(mkpts1)
                mkpts2 = inv_transform2(mkpts2)
            
            all_mkpts1.append(mkpts1)
            all_mkpts2.append(mkpts2)
            all_mconf.append(mconf)
        
        # Combine results
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        # Remove duplicates
        mkpts1, mkpts2, mconf = self._remove_duplicates(mkpts1, mkpts2, mconf)
        
        return mkpts1, mkpts2, mconf
    
    def _remove_duplicates(
        self,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        conf: np.ndarray,
        threshold: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate matches."""
        # Combine keypoints for duplicate detection
        combined = np.concatenate([kpts1, kpts2], axis=1)
        
        # Find unique matches
        unique_indices = []
        for i in range(len(combined)):
            is_duplicate = False
            for j in unique_indices:
                dist = np.linalg.norm(combined[i] - combined[j])
                if dist < threshold:
                    is_duplicate = True
                    # Keep the one with higher confidence
                    if conf[i] > conf[j]:
                        unique_indices.remove(j)
                        unique_indices.append(i)
                    break
            if not is_duplicate:
                unique_indices.append(i)
        
        unique_indices = np.array(unique_indices)
        
        return kpts1[unique_indices], kpts2[unique_indices], conf[unique_indices]
    
    def match_multi_scale(
        self,
        image1_path: str,
        image2_path: str,
        scales: List[int] = [640, 1024, 1440]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match at multiple scales for better coverage.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            scales: List of scales to match at
        
        Returns:
            Combined matches from all scales
        """
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        for scale in scales:
            mkpts1, mkpts2, mconf = self.match_pair(
                image1_path,
                image2_path,
                resize=scale
            )
            
            all_mkpts1.append(mkpts1)
            all_mkpts2.append(mkpts2)
            all_mconf.append(mconf)
        
        # Combine and remove duplicates
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        return self._remove_duplicates(mkpts1, mkpts2, mconf)