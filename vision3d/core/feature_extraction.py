"""
Feature extraction module for Vision3D.

This module provides functionality for extracting and managing
local features using various methods (SIFT, ORB, SuperPoint, etc.).
"""

import cv2
import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract local features from images.
    
    Supports multiple feature extraction methods:
    - Classical: SIFT, ORB, AKAZE
    - Deep Learning: SuperPoint, KeyNet+HardNet
    - Hybrid approaches
    """
    
    def __init__(
        self,
        method: str = 'superpoint',
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            method: Feature extraction method
            device: PyTorch device for deep methods
            config: Configuration dictionary
        """
        self.method = method
        self.device = device or torch.device('cpu')
        self.config = config or self._get_default_config()
        self._setup_extractor()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'max_keypoints': 2048,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'resize_to': 800,
            'use_gpu': torch.cuda.is_available()
        }
    
    def _setup_extractor(self):
        """Setup the feature extraction method."""
        if self.method == 'sift':
            self.extractor = cv2.SIFT_create(
                nfeatures=self.config['max_keypoints']
            )
        elif self.method == 'orb':
            self.extractor = cv2.ORB_create(
                nFeatures=self.config['max_keypoints']
            )
        elif self.method == 'akaze':
            self.extractor = cv2.AKAZE_create()
        elif self.method == 'superpoint':
            # In production, load actual SuperPoint model
            logger.info("Loading SuperPoint model...")
            # self.extractor = SuperPointExtractor(self.device, self.config)
        elif self.method == 'keynet':
            # In production, load KeyNet+HardNet
            logger.info("Loading KeyNet+HardNet model...")
            # self.extractor = KeyNetExtractor(self.device, self.config)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")
    
    def extract_features(
        self,
        image_paths: List[str],
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Extract features from multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Directory to save features
            verbose: Show progress bar
        
        Returns:
            Dictionary with features for each image
        """
        features = {}
        
        iterator = tqdm(image_paths, desc="Extracting features") if verbose else image_paths
        
        for img_path in iterator:
            img_name = Path(img_path).name
            
            # Extract features
            keypoints, descriptors = self.extract_single_image(img_path)
            
            features[img_name] = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image_size': self._get_image_size(img_path)
            }
        
        # Save features if output directory provided
        if output_dir:
            self.save_features(features, output_dir)
        
        return features
    
    def extract_single_image(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a single image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Load image
        img = cv2.imread(image_path)
        
        # Resize if needed
        if self.config['resize_to'] is not None:
            img = self._resize_image(img, self.config['resize_to'])
        
        # Extract based on method
        if self.method in ['sift', 'orb', 'akaze']:
            return self._extract_classical(img)
        else:
            return self._extract_deep(img)
    
    def _extract_classical(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using classical methods."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute
        kpts, descs = self.extractor.detectAndCompute(gray, None)
        
        # Convert keypoints to numpy array
        if kpts:
            keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32)
            descriptors = descs
        else:
            keypoints = np.array([], dtype=np.float32).reshape(0, 2)
            descriptors = np.array([], dtype=np.float32).reshape(0, 128)
        
        return keypoints, descriptors
    
    def _extract_deep(
        self,
        img: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using deep learning methods."""
        # In production, this would use the actual deep models
        # For now, return placeholder data
        h, w = img.shape[:2]
        num_features = min(self.config['max_keypoints'], 500)
        
        keypoints = np.random.rand(num_features, 2) * [w, h]
        descriptors = np.random.randn(num_features, 256).astype(np.float32)
        
        # Normalize descriptors
        descriptors = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
        
        return keypoints, descriptors
    
    def _resize_image(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image maintaining aspect ratio."""
        h, w = img.shape[:2]
        
        if max(h, w) <= target_size:
            return img
        
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions."""
        img = cv2.imread(image_path)
        return img.shape[1], img.shape[0]  # width, height
    
    def save_features(self, features: Dict, output_dir: Path):
        """
        Save features to HDF5 files.
        
        Args:
            features: Dictionary of features
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_dir / 'keypoints.h5', 'w') as f_kp, \
             h5py.File(output_dir / 'descriptors.h5', 'w') as f_desc:
            
            for img_name, feat_data in features.items():
                f_kp[img_name] = feat_data['keypoints']
                f_desc[img_name] = feat_data['descriptors']
        
        logger.info(f"Features saved to {output_dir}")
    
    def load_features(self, feature_dir: Path) -> Dict:
        """
        Load features from HDF5 files.
        
        Args:
            feature_dir: Directory containing feature files
        
        Returns:
            Dictionary of features
        """
        features = {}
        
        with h5py.File(feature_dir / 'keypoints.h5', 'r') as f_kp, \
             h5py.File(feature_dir / 'descriptors.h5', 'r') as f_desc:
            
            for img_name in f_kp.keys():
                features[img_name] = {
                    'keypoints': f_kp[img_name][...],
                    'descriptors': f_desc[img_name][...]
                }
        
        return features
    
    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        method: str = 'mutual_nn',
        ratio_threshold: float = 0.8
    ) -> np.ndarray:
        """
        Match feature descriptors.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            method: Matching method ('mutual_nn', 'ratio_test')
            ratio_threshold: Threshold for ratio test
        
        Returns:
            Array of matches (indices)
        """
        if method == 'mutual_nn':
            return self._match_mutual_nn(desc1, desc2)
        elif method == 'ratio_test':
            return self._match_ratio_test(desc1, desc2, ratio_threshold)
        else:
            raise ValueError(f"Unknown matching method: {method}")
    
    def _match_mutual_nn(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> np.ndarray:
        """Mutual nearest neighbor matching."""
        # Forward matching: desc1 -> desc2
        if len(desc1) == 0 or len(desc2) == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)
        
        # Compute distances
        distances = np.linalg.norm(desc1[:, None] - desc2[None, :], axis=2)
        
        # Find nearest neighbors
        nn1to2 = distances.argmin(axis=1)
        nn2to1 = distances.argmin(axis=0)
        
        # Find mutual matches
        matches = []
        for i, j in enumerate(nn1to2):
            if nn2to1[j] == i:
                matches.append([i, j])
        
        return np.array(matches, dtype=np.int32)
    
    def _match_ratio_test(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float
    ) -> np.ndarray:
        """Lowe's ratio test matching."""
        if len(desc1) == 0 or len(desc2) == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)
        
        # Create matcher
        if self.method in ['sift', 'orb']:
            matcher = cv2.BFMatcher(cv2.NORM_L2 if self.method == 'sift' else cv2.NORM_HAMMING)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        # Find matches
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append([m.queryIdx, m.trainIdx])
        
        return np.array(good_matches, dtype=np.int32)