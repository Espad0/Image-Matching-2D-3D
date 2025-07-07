"""
Base matcher interface for all feature matching models.

This module defines the common interface that all matchers (SuperGlue, LoFTR, etc.)
must implement, ensuring consistency and modularity in the matching pipeline.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class BaseMatcher(ABC):
    """
    Abstract base class for all feature matchers.
    
    This class defines the interface that all matchers must implement:
    - match_pair: Match features between two images
    - match_pairs: Match features for multiple image pairs
    - preprocess_image: Prepare images for matching
    
    Attributes:
        device: PyTorch device for computation
        config: Configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        """
        Initialize the matcher.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device (cuda/cpu)
        """
        self.config = config or self.get_default_config()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized {self.__class__.__name__} on device: {self.device}")
    
    @abstractmethod
    def get_default_config(self) -> Dict:
        """Get default configuration for the matcher."""
        pass
    
    @abstractmethod
    def match_pair(self, image1_path: str, image2_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match features between two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            **kwargs: Additional matcher-specific arguments
            
        Returns:
            kpts1: Keypoints in image 1 (N x 2)
            kpts2: Keypoints in image 2 (N x 2)
            confidence: Match confidence scores (N,)
        """
        pass
    
    def match_pairs(self, image_paths: List[str], pairs: List[Tuple[int, int]], 
                   output_dir: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Match features for multiple image pairs.
        
        Args:
            image_paths: List of image paths
            pairs: List of (idx1, idx2) pairs to match
            output_dir: Optional directory to save matches
            verbose: Show progress
            
        Returns:
            Dictionary mapping (idx1, idx2) to match data
        """
        matches = {}
        
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(pairs, desc=f"Matching with {self.__class__.__name__}")
        else:
            iterator = pairs
        
        for idx1, idx2 in iterator:
            try:
                # Get image paths
                img1_path = image_paths[idx1]
                img2_path = image_paths[idx2]
                
                # Match the pair
                kpts1, kpts2, confidence = self.match_pair(img1_path, img2_path)
                
                # Store matches if any found
                if len(kpts1) > 0:
                    matches[(idx1, idx2)] = {
                        'keypoints1': kpts1,
                        'keypoints2': kpts2,
                        'confidence': confidence,
                        'num_matches': len(kpts1)
                    }
                    logger.debug(f"Matched {img1_path} - {img2_path}: {len(kpts1)} matches")
                else:
                    logger.warning(f"No matches found for {img1_path} - {img2_path}")
                    
            except Exception as e:
                logger.error(f"Error matching pair ({idx1}, {idx2}): {e}")
                continue
        
        logger.info(f"Matched {len(matches)} pairs out of {len(pairs)} total pairs")
        return matches
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, float]:
        """
        Preprocess image for matching.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional preprocessing parameters
            
        Returns:
            Preprocessed image and scale factor
        """
        pass
    
    def visualize_matches(self, img1: np.ndarray, img2: np.ndarray,
                         kpts1: np.ndarray, kpts2: np.ndarray,
                         confidence: Optional[np.ndarray] = None,
                         output_path: Optional[str] = None):
        """
        Visualize matches between two images.
        
        Args:
            img1, img2: Input images
            kpts1, kpts2: Matched keypoints
            confidence: Match confidence scores
            output_path: Optional path to save visualization
        """
        import cv2
        import matplotlib.pyplot as plt
        
        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:] = img2
        
        # Draw matches
        for i, (pt1, pt2) in enumerate(zip(kpts1, kpts2)):
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2 + np.array([w1, 0])))
            
            # Color based on confidence if available
            if confidence is not None:
                color = plt.cm.plasma(confidence[i])[:3]
                color = tuple(int(c * 255) for c in color)
            else:
                color = (0, 255, 0)
            
            cv2.line(canvas, pt1, pt2, color, 1)
            cv2.circle(canvas, pt1, 3, color, -1)
            cv2.circle(canvas, pt2, 3, color, -1)
        
        if output_path:
            cv2.imwrite(output_path, canvas)
        
        return canvas