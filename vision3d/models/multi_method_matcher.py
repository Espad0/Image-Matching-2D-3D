"""
Multi-method matcher that combines multiple matching algorithms.

Based on the IMC-2023 winning solution, this matcher combines:
- LoFTR for dense matching (good for textureless regions)
- SuperGlue for sparse matching (good for distinctive features)

The combination gives us the best of both worlds!
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from .loftr import LoFTRMatcher
from .superglue import SuperGlueMatcher
from .base import BaseMatcher

logger = logging.getLogger(__name__)


class MultiMethodMatcher(BaseMatcher):
    """
    Multi-method matcher that combines multiple matching algorithms.
    
    This class implements the IMC-2023 winning strategy of combining
    different matchers to get more and better quality matches.
    """
    
    def __init__(
        self,
        device: torch.device,
        config: Optional[Dict] = None
    ):
        """
        Initialize multi-method matcher.
        
        Args:
            device: PyTorch device for computation
            config: Configuration dictionary
        """
        super().__init__(device, config or self._get_default_config())
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'methods': ['loftr', 'superglue'],
            'adaptive_threshold': 400,  # Number of image pairs for adaptive strategy
            'loftr_config': {
                'image_resize': 1024,
                'multi_resolution': [1024, 1440],
                'confidence_threshold': 0.3,
                'use_tta': True,
                'tta_variants': ['orig', 'flip_lr']
            },
            'superglue_config': {
                'max_keypoints': 8096,
                'keypoint_threshold': 0.005,
                'nms_radius': 4,
                'match_threshold': 0.2,
                'use_tta': True,
                'tta_groups': [('orig', 'orig'), ('flip_lr', 'flip_lr')]
            },
            'dedup_threshold': 3.0
        }
    
    def _setup(self):
        """Setup the matchers."""
        logger.info("Setting up multi-method matcher...")
        
        # Initialize LoFTR
        if 'loftr' in self.config['methods']:
            # Merge with LoFTR defaults
            loftr_config = self.config['loftr_config'].copy()
            self.loftr = LoFTRMatcher(
                device=self.device,
                config=loftr_config
            )
            self.loftr._setup()
        
        # Initialize SuperGlue
        if 'superglue' in self.config['methods']:
            # Get SuperGlue default config and update with our config
            from .superglue import SuperGlueMatcher
            sg_matcher_temp = SuperGlueMatcher(self.device)
            sg_config = sg_matcher_temp._get_default_config()
            sg_config.update(self.config['superglue_config'])
            
            self.superglue = SuperGlueMatcher(
                device=self.device,
                config=sg_config
            )
            self.superglue._setup()
        
        logger.info(f"Multi-method matcher ready with methods: {self.config['methods']}")
    
    def match_pair(
        self,
        image1_path: str,
        image2_path: str,
        num_pairs: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find matching points between two images using multiple methods.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            num_pairs: Number of total image pairs in the scene (for adaptive strategy)
        
        Returns:
            Tuple of (mkpts1, mkpts2, mconf) with combined matches
        """
        logger.info(f"Multi-method matching: {Path(image1_path).name} <-> {Path(image2_path).name}")
        
        all_mkpts1 = []
        all_mkpts2 = []
        all_mconf = []
        
        # Adaptive strategy based on scene size (IMC-2023 approach)
        if num_pairs and num_pairs >= self.config['adaptive_threshold']:
            # Large scenes: Use only SuperGlue with more keypoints
            logger.info(f"Large scene ({num_pairs} pairs): Using SuperGlue only")
            methods_to_use = ['superglue'] if 'superglue' in self.config['methods'] else []
        else:
            # Small scenes: Use all methods
            methods_to_use = self.config['methods']
            logger.info(f"Small scene: Using all methods {methods_to_use}")
        
        # Run LoFTR if enabled
        if 'loftr' in methods_to_use and hasattr(self, 'loftr'):
            logger.debug("Running LoFTR matching...")
            
            # Use multi-resolution + bidirectional for best results
            mkpts1, mkpts2, mconf = self.loftr.match_pair_multires(
                image1_path, image2_path,
                resolutions=self.config['loftr_config'].get('multi_resolution', [1024, 1440]),
                bidirectional=True
            )
            
            if len(mkpts1) > 0:
                all_mkpts1.append(mkpts1)
                all_mkpts2.append(mkpts2)
                all_mconf.append(mconf)
                logger.info(f"LoFTR: {len(mkpts1)} matches")
        
        # Run SuperGlue if enabled
        if 'superglue' in methods_to_use and hasattr(self, 'superglue'):
            logger.debug("Running SuperGlue matching...")
            
            try:
                mkpts1, mkpts2, mconf = self.superglue.match_pair(
                    image1_path, image2_path
                )
                
                if len(mkpts1) > 0:
                    all_mkpts1.append(mkpts1)
                    all_mkpts2.append(mkpts2)
                    all_mconf.append(mconf)
                    logger.info(f"SuperGlue: {len(mkpts1)} matches")
            except Exception as e:
                logger.warning(f"SuperGlue matching failed: {e}. Continuing with LoFTR only.")
        
        # Combine all matches
        if not all_mkpts1:
            logger.warning("No matches found from any method")
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([])
        
        mkpts1 = np.concatenate(all_mkpts1, axis=0)
        mkpts2 = np.concatenate(all_mkpts2, axis=0)
        mconf = np.concatenate(all_mconf, axis=0)
        
        logger.info(f"Combined: {len(mkpts1)} matches before deduplication")
        
        # Advanced deduplication
        mkpts1, mkpts2, mconf = self._advanced_deduplication(
            mkpts1, mkpts2, mconf
        )
        
        logger.info(f"Final: {len(mkpts1)} matches after deduplication")
        
        return mkpts1, mkpts2, mconf
    
    def _advanced_deduplication(
        self,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        conf: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advanced deduplication for multi-method matches.
        
        This is more sophisticated than simple spatial deduplication because
        matches from different methods might have slightly different coordinates
        for the same feature.
        """
        if len(kpts1) == 0:
            return kpts1, kpts2, conf
        
        if threshold is None:
            threshold = self.config['dedup_threshold']
        
        logger.debug(f"Advanced deduplication: {len(kpts1)} matches, threshold={threshold}")
        
        try:
            from scipy.spatial import cKDTree
            
            # Combine coordinates for matching
            combined = np.concatenate([kpts1, kpts2], axis=1)
            
            # Sort by confidence (keep highest confidence matches)
            sorted_idx = np.argsort(conf)[::-1]
            
            # Build KD-tree for efficient nearest neighbor search
            tree = cKDTree(combined[sorted_idx])
            
            # Find groups of nearby matches
            groups = tree.query_ball_tree(tree, threshold)
            
            # Keep only the highest confidence match from each group
            keep_mask = np.zeros(len(sorted_idx), dtype=bool)
            seen = set()
            
            for i, group in enumerate(groups):
                if i not in seen:
                    # Keep this match (highest confidence in group)
                    keep_mask[i] = True
                    # Mark all others in group as seen
                    seen.update(group)
            
            # Apply mask to original (sorted) indices
            keep_indices = sorted_idx[keep_mask]
            
            logger.debug(f"Kept {len(keep_indices)} unique matches")
            
            return kpts1[keep_indices], kpts2[keep_indices], conf[keep_indices]
            
        except ImportError:
            logger.warning("scipy not available, using simple deduplication")
            # Fallback to simple deduplication
            return self._simple_deduplication(kpts1, kpts2, conf, threshold)
    
    def _simple_deduplication(
        self,
        kpts1: np.ndarray,
        kpts2: np.ndarray,
        conf: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple deduplication without scipy."""
        # Sort by confidence
        sorted_idx = np.argsort(conf)[::-1]
        
        # Keep top N matches
        max_matches = 10000
        if len(sorted_idx) > max_matches:
            sorted_idx = sorted_idx[:max_matches]
        
        return kpts1[sorted_idx], kpts2[sorted_idx], conf[sorted_idx]