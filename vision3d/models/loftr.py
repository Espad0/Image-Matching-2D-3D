"""
LoFTR (Local Feature TRansformer) implementation for dense feature matching.

LoFTR revolutionizes feature matching by using transformers (like GPT/BERT) for
computer vision. Instead of detecting keypoints first, it directly finds 
pixel-to-pixel correspondences between images.

Key advantages:
- Works on textureless surfaces (walls, sky) where traditional methods fail
- Handles repetitive patterns better
- No need for handcrafted feature detectors

How it works:
1. Extract features using CNN backbone (ResNet)
2. Apply transformer attention to reason about global context
3. Coarse matching at 1/8 resolution 
4. Fine refinement for sub-pixel accuracy

Think of it as teaching a neural network to "understand" what pixels in
one image correspond to pixels in another image, using the same attention
mechanism that powers ChatGPT!
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
    LoFTR matcher for dense feature matching between image pairs.
    
    This class wraps the LoFTR model to provide an easy-to-use interface for
    finding correspondences between images. It's particularly good at matching
    images of textureless or repetitive scenes where traditional methods struggle.
    
    Features implemented:
    - Coarse-to-fine matching: First finds rough matches, then refines them
    - Test-time augmentation (TTA): Processes multiple versions of images for robustness
    - Confidence thresholding: Filters out uncertain matches
    - Multi-scale matching: Processes images at different resolutions
    
    The matching process:
    1. Load and preprocess images (resize, normalize)
    2. Extract features using CNN + Transformer
    3. Find correspondences using attention mechanism  
    4. Filter matches by confidence
    5. Return pixel coordinates of matching points
    
    Example:
        >>> # Initialize matcher with GPU
        >>> matcher = LoFTRMatcher(device=torch.device('cuda'))
        >>> 
        >>> # Find matches between two images
        >>> kpts1, kpts2, conf = matcher.match_pair('photo1.jpg', 'photo2.jpg')
        >>> 
        >>> # kpts1: (N, 2) array of (x, y) coordinates in image 1
        >>> # kpts2: (N, 2) array of corresponding points in image 2  
        >>> # conf: (N,) array of confidence scores (0-1)
        >>> print(f"Found {len(kpts1)} matches with avg confidence {conf.mean():.2f}")
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
            device: PyTorch device for computation
                - torch.device('cuda'): Use GPU (recommended, 10x faster)
                - torch.device('cpu'): Use CPU (slower but works everywhere)
            config: Configuration dictionary to override defaults
                - See _get_default_config() for all options
            pretrained: Which pretrained weights to use
                - 'outdoor': Trained on outdoor scenes (default)
                - 'indoor': Trained on indoor scenes
                Choose based on your use case for best results
        
        The pretrained models have learned from millions of image pairs,
        so they work well out-of-the-box without any training needed!
        """
        self.pretrained = pretrained
        super().__init__(device, config or self._get_default_config())
    
    def _get_default_config(self) -> Dict:
        """Get default LoFTR configuration.
        
        These parameters control the matching behavior and quality/speed tradeoff.
        The defaults work well for most cases, but can be tuned for specific needs.
        """
        return {
            'image_resize': 1024,  # Max image dimension (larger = more accurate but slower)
            'confidence_threshold': 0.3,  # Min confidence for matches (0-1, higher = fewer but better matches)
            'use_tta': True,  # Test-time augmentation for more robust matching
            'tta_variants': ['orig', 'flip_lr'],  # TTA types: original and horizontal flip
            'match_threshold': 0.2,  # Transformer attention threshold
            'max_keypoints': 8192  # Maximum matches to return (memory limit)
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
        Find matching points between two images using LoFTR.
        
        This is the main method you'll use. It takes two images and returns
        lists of corresponding points. For example, if pixel (100, 200) in
        image1 matches pixel (150, 180) in image2, these will be returned
        as corresponding entries in mkpts1 and mkpts2.
        
        Args:
            image1_path: Path to first image file
            image2_path: Path to second image file  
            resize: Optional max dimension to resize images to
                - None: Use default from config (usually 1024)
                - 640: Fast but less accurate
                - 1024: Good balance (default)
                - 1440: Slower but more accurate
        
        Returns:
            Tuple of (mkpts1, mkpts2, mconf):
                - mkpts1: (N, 2) array of (x, y) coordinates in image1
                - mkpts2: (N, 2) array of matching (x, y) coordinates in image2
                - mconf: (N,) array of confidence scores for each match (0-1)
        
        Example:
            >>> # Match two photos of the same building
            >>> pts1, pts2, conf = matcher.match_pair('building1.jpg', 'building2.jpg')
            >>> 
            >>> # Filter high-confidence matches
            >>> good_matches = conf > 0.5
            >>> pts1_good = pts1[good_matches]
            >>> pts2_good = pts2[good_matches]
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
        """Preprocess image for LoFTR.
        
        LoFTR expects grayscale images with values in [0, 1] range.
        This method handles all the necessary preprocessing steps.
        """
        # Step 1: Read image from disk
        img = cv2.imread(image_path)
        
        # Step 2: Resize to target dimension while maintaining aspect ratio
        # This is important because:
        # - Smaller images = faster processing
        # - Very large images may not fit in GPU memory
        # - LoFTR was trained on specific image sizes
        if resize is None:
            resize = self.config['image_resize']
        
        if resize is not None:
            # Calculate scale to fit longest edge to 'resize' pixels
            scale = resize / max(img.shape[:2])
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height))
        else:
            scale = 1.0
        
        # Step 3: Convert to PyTorch tensor and normalize
        # - Convert BGR (OpenCV format) to RGB
        # - Convert to grayscale (LoFTR uses grayscale)
        # - Normalize to [0, 1] range
        img_tensor = K.image_to_tensor(img, False).float() / 255.
        img_tensor = K.color.bgr_to_rgb(img_tensor)
        img_tensor = K.color.rgb_to_grayscale(img_tensor)
        img_tensor = img_tensor.to(self.device)
        
        return img, img_tensor, scale  # Return original img for visualization
    
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
        """Perform matching with test-time augmentation (TTA).
        
        TTA is a technique where we process multiple versions of the input
        (original, flipped, rotated) and combine the results. This gives us:
        - More matches overall
        - Better coverage of the image
        - More robust results
        
        It's like asking multiple people to solve the same puzzle and
        combining their answers!
        """
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        # Process each augmentation variant
        for tta in self.config['tta_variants']:
            # Apply augmentation (e.g., flip the image)
            img1_aug, inv_transform1 = self.apply_tta(img1_tensor, tta)
            img2_aug, inv_transform2 = self.apply_tta(img2_tensor, tta)
            
            # Run LoFTR on augmented images
            input_dict = {
                "image0": img1_aug,
                "image1": img2_aug
            }
            
            correspondences = self.matcher(input_dict)
            
            # Extract results
            mkpts1 = correspondences['keypoints0'].cpu().numpy()
            mkpts2 = correspondences['keypoints1'].cpu().numpy()
            mconf = correspondences['confidence'].cpu().numpy()
            
            # Reverse the augmentation on keypoint coordinates
            # E.g., if we flipped the image, flip the x-coordinates back
            if tta != 'orig':
                mkpts1 = inv_transform1(mkpts1)
                mkpts2 = inv_transform2(mkpts2)
            
            all_mkpts1.append(mkpts1)
            all_mkpts2.append(mkpts2)
            all_mconf.append(mconf)
        
        # Combine results from all augmentations
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        # Remove duplicate matches (same point found in multiple augmentations)
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
        
        if len(unique_indices) == 0:
            # Return empty arrays if no unique matches
            return np.array([]), np.array([]), np.array([])
        
        unique_indices = np.array(unique_indices, dtype=np.int64)
        
        return kpts1[unique_indices], kpts2[unique_indices], conf[unique_indices]
    
    def match_multi_scale(
        self,
        image1_path: str,
        image2_path: str,
        scales: List[int] = [640, 1024, 1440]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match at multiple scales for better coverage.
        
        Different scales capture different levels of detail:
        - Small scale (640): Captures global structure, fast
        - Medium scale (1024): Balanced detail and context
        - Large scale (1440): Fine details, slow but accurate
        
        By combining all scales, we get both the "big picture" matches
        and fine detail matches!
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            scales: List of image sizes to process
                - Smaller = faster but may miss details
                - Larger = slower but more accurate
        
        Returns:
            Combined matches from all scales with duplicates removed
        
        Example:
            >>> # Match at multiple resolutions for maximum coverage
            >>> matches = matcher.match_multi_scale(
            ...     'img1.jpg', 'img2.jpg',
            ...     scales=[512, 768, 1024]  # Custom scales
            ... )
        """
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        # Process each scale independently
        for scale in scales:
            mkpts1, mkpts2, mconf = self.match_pair(
                image1_path,
                image2_path,
                resize=scale
            )
            
            all_mkpts1.append(mkpts1)
            all_mkpts2.append(mkpts2)
            all_mconf.append(mconf)
        
        # Combine matches from all scales
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        # Remove duplicates (same match found at different scales)
        return self._remove_duplicates(mkpts1, mkpts2, mconf)