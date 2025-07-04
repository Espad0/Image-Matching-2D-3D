"""
Improvements for Vision3D pipeline based on IMC notebook analysis.
"""

import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedLoFTRMatcher:
    """Enhanced LoFTR matcher with multi-resolution and TTA support."""
    
    def __init__(self, matcher, config):
        self.matcher = matcher
        self.config = config
        
    def match_pair_multires(
        self,
        image1_path: str,
        image2_path: str,
        resolutions: List[int] = [1024, 1440],
        tta: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match image pair at multiple resolutions with TTA.
        
        Key improvements:
        1. Multiple resolutions for better accuracy
        2. Test-time augmentation (horizontal flip)
        3. Bidirectional matching
        """
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        for resolution in resolutions:
            # Forward matching
            mkpts1, mkpts2, mconf = self.matcher.match_pair(
                image1_path, image2_path, resize=resolution
            )
            all_mkpts1.append(mkpts1)
            all_mkpts2.append(mkpts2)
            all_mconf.append(mconf)
            
            if tta:
                # Horizontal flip augmentation
                mkpts1_flip, mkpts2_flip, mconf_flip = self._match_with_flip(
                    image1_path, image2_path, resolution
                )
                all_mkpts1.append(mkpts1_flip)
                all_mkpts2.append(mkpts2_flip)
                all_mconf.append(mconf_flip)
        
        # Bidirectional matching (swap image order)
        mkpts2_rev, mkpts1_rev, mconf_rev = self.matcher.match_pair(
            image2_path, image1_path, resize=resolutions[0]
        )
        all_mkpts1.append(mkpts1_rev)
        all_mkpts2.append(mkpts2_rev)
        all_mconf.append(mconf_rev)
        
        # Concatenate all matches
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        # Remove duplicates
        mkpts1, mkpts2, mconf = self._deduplicate_matches(mkpts1, mkpts2, mconf)
        
        return mkpts1, mkpts2, mconf
    
    def _match_with_flip(
        self,
        image1_path: str,
        image2_path: str,
        resolution: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Match with horizontal flip augmentation."""
        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Flip images
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)
        
        # Save temporary flipped images
        temp_dir = Path("/tmp/vision3d_tta")
        temp_dir.mkdir(exist_ok=True)
        
        flip1_path = temp_dir / "img1_flip.jpg"
        flip2_path = temp_dir / "img2_flip.jpg"
        cv2.imwrite(str(flip1_path), img1_flip)
        cv2.imwrite(str(flip2_path), img2_flip)
        
        # Match flipped images
        mkpts1, mkpts2, mconf = self.matcher.match_pair(
            str(flip1_path), str(flip2_path), resize=resolution
        )
        
        # Unflip keypoints
        mkpts1[:, 0] = w1 - mkpts1[:, 0] - 1
        mkpts2[:, 0] = w2 - mkpts2[:, 0] - 1
        
        # Clean up
        flip1_path.unlink()
        flip2_path.unlink()
        
        return mkpts1, mkpts2, mconf
    
    def _deduplicate_matches(
        self,
        mkpts1: np.ndarray,
        mkpts2: np.ndarray,
        mconf: np.ndarray,
        threshold: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate matches based on distance threshold."""
        if len(mkpts1) == 0:
            return mkpts1, mkpts2, mconf
            
        # Concatenate points for duplicate detection
        points = np.concatenate([mkpts1, mkpts2], axis=1)
        
        # Find unique matches
        unique_indices = []
        for i in range(len(points)):
            is_unique = True
            for j in unique_indices:
                # Check if points are too close
                dist = np.linalg.norm(points[i] - points[j])
                if dist < threshold:
                    # Keep the one with higher confidence
                    if mconf[i] > mconf[j]:
                        unique_indices.remove(j)
                        unique_indices.append(i)
                    is_unique = False
                    break
            if is_unique:
                unique_indices.append(i)
        
        unique_indices = np.array(unique_indices)
        return mkpts1[unique_indices], mkpts2[unique_indices], mconf[unique_indices]


def create_improved_pipeline_config():
    """Create configuration for improved Vision3D pipeline."""
    return {
        'matcher': {
            'resolutions': [1024, 1440],  # Multi-resolution matching
            'use_tta': True,               # Test-time augmentation
            'confidence_threshold': 0.3,    # Higher confidence threshold
            'image_resize': 1024,          # Default higher resolution
        },
        'reconstruction': {
            'mapper_options': {
                'min_model_size': 3,
                'ba_local_max_refinements': 2,
                'ba_global_max_refinements': 20,
                'ba_global_max_num_iterations': 50,
                'ba_local_max_num_iterations': 25,
                'init_max_error': 4.0,  # Pixels
                'filter_max_reproj_error': 4.0,
                'filter_min_tri_angle': 1.5,  # Degrees
            }
        },
        'pair_selection': {
            'use_global_descriptors': True,
            'similarity_threshold': 0.5,
            'min_pairs_per_image': 35,
            'exhaustive_if_less': 20,
        }
    }


def apply_mapper_options(options_dict: Dict) -> 'pycolmap.IncrementalMapperOptions':
    """Apply custom mapper options for COLMAP reconstruction."""
    import pycolmap
    
    options = pycolmap.IncrementalMapperOptions()
    
    # Apply each option if it exists
    option_mapping = {
        'min_model_size': 'min_model_size',
        'ba_local_max_refinements': 'ba_local_max_refinements',
        'ba_global_max_refinements': 'ba_global_max_refinements',
        'ba_global_max_num_iterations': 'ba_global_max_num_iterations',
        'ba_local_max_num_iterations': 'ba_local_max_num_iterations',
        'init_max_error': 'init_max_error',
        'filter_max_reproj_error': 'filter_max_reproj_error',
        'filter_min_tri_angle': 'filter_min_tri_angle',
    }
    
    for key, attr in option_mapping.items():
        if key in options_dict:
            setattr(options, attr, options_dict[key])
    
    return options


class HybridMatcher:
    """Combine LoFTR and SuperGlue for robust matching."""
    
    def __init__(self, loftr_matcher, superglue_matcher=None):
        self.loftr = loftr_matcher
        self.superglue = superglue_matcher
        
    def match_pair(
        self,
        image1_path: str,
        image2_path: str,
        num_pairs: int = 400
    ) -> Dict:
        """
        Match using appropriate method based on scene size.
        
        IMC strategy:
        - >= 400 pairs: SuperGlue only
        - < 400 pairs: LoFTR + SuperGlue combined
        """
        if num_pairs >= 400 and self.superglue is not None:
            # Large scene: SuperGlue only
            return self._match_superglue_multires(image1_path, image2_path)
        else:
            # Small scene: Combine LoFTR + SuperGlue
            return self._match_combined(image1_path, image2_path)
    
    def _match_superglue_multires(
        self,
        image1_path: str,
        image2_path: str
    ) -> Dict:
        """Match using SuperGlue at multiple resolutions."""
        # Implementation would call SuperGlue matcher
        # This is a placeholder
        pass
    
    def _match_combined(
        self,
        image1_path: str,
        image2_path: str  
    ) -> Dict:
        """Combine LoFTR and SuperGlue matches."""
        # Get LoFTR matches at multiple resolutions
        mkpts1_loftr, mkpts2_loftr, conf_loftr = self.loftr.match_pair_multires(
            image1_path, image2_path,
            resolutions=[1024],
            tta=True
        )
        
        # Get SuperGlue matches if available
        if self.superglue is not None:
            # Combine with SuperGlue matches
            # This is a placeholder for actual implementation
            pass
        
        return {
            'keypoints1': mkpts1_loftr,
            'keypoints2': mkpts2_loftr,
            'confidence': conf_loftr
        }