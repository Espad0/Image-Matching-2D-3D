"""
Feature extraction and matching orchestration.

This module coordinates the feature matching process, including:
- Selecting appropriate matcher based on dataset size
- Managing batch processing
- Handling multi-scale matching
- Saving results to h5 files
"""

import os
import h5py
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from copy import deepcopy
from fastprogress import progress_bar

from ..models.superglue_matcher import SuperGlueMatcher
from ..models.loftr import LoFTRMatcher
from ..utils.colmap_interface import process_matches_to_unique_keypoints

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Orchestrates feature extraction and matching for 3D reconstruction.
    
    This class handles:
    - Automatic matcher selection based on dataset size
    - Multi-method matching (LoFTR + SuperGlue)
    - Batch processing with progress tracking
    - Result aggregation and saving
    """
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[torch.device] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
        """
        self.config = config or self._get_default_config()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._matchers = {}
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'min_matches': 15,
            'large_dataset_threshold': 400,
            'loftr_input_size': 1024,
            'superglue_sizes': [1024, 1440],
            'high_res_keypoints': 8192,
            'output_format': 'h5'
        }
    
    def _get_matcher(self, matcher_type: str):
        """Get or create a matcher instance."""
        if matcher_type not in self._matchers:
            if matcher_type == 'loftr':
                self._matchers[matcher_type] = LoFTRMatcher(device=self.device)
            elif matcher_type == 'superglue':
                self._matchers[matcher_type] = SuperGlueMatcher(device=self.device)
            elif matcher_type == 'superglue_high_res':
                # Create high-resolution SuperGlue config
                config = deepcopy(SuperGlueMatcher().get_default_config())
                config['superpoint']['max_keypoints'] = self.config['high_res_keypoints']
                self._matchers[matcher_type] = SuperGlueMatcher(config=config, device=self.device)
            else:
                raise ValueError(f"Unknown matcher type: {matcher_type}")
        
        return self._matchers[matcher_type]
    
    def match_images(
        self,
        img_filenames: List[str],
        index_pairs: List[Tuple[int, int]],
        method: str = 'auto',
        output_dir: str = './features',
        verbose: bool = True
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Match images using the specified method.
        
        Args:
            img_filenames: List of image file paths
            index_pairs: List of (idx1, idx2) pairs to match
            method: Matching method ('loftr', 'superglue', 'loftr_superglue', 'auto')
            output_dir: Directory to save features
            verbose: Show progress
            
        Returns:
            Dictionary of matches
        """
        # Auto-select method based on dataset size
        if method == 'auto':
            if len(index_pairs) >= self.config['large_dataset_threshold']:
                method = 'superglue'
                logger.info(f"Large dataset ({len(index_pairs)} pairs), using SuperGlue only")
            else:
                method = 'loftr_superglue'
                logger.info(f"Small dataset ({len(index_pairs)} pairs), using LoFTR + SuperGlue")
        
        # Run matching
        if method == 'loftr_superglue':
            matches = self._match_loftr_superglue(img_filenames, index_pairs, verbose)
        elif method == 'superglue':
            matches = self._match_superglue_only(img_filenames, index_pairs, verbose)
        elif method == 'loftr':
            matches = self._match_loftr_only(img_filenames, index_pairs, verbose)
        else:
            raise ValueError(f"Unknown matching method: {method}")
        
        # Save results
        if output_dir:
            self._save_matches(matches, img_filenames, output_dir)
        
        return matches
    
    def _match_loftr_superglue(
        self,
        img_filenames: List[str],
        index_pairs: List[Tuple[int, int]],
        verbose: bool
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Match using both LoFTR and SuperGlue for maximum robustness.
        """
        logger.info(f"Matching {len(index_pairs)} pairs using LoFTR + SuperGlue")
        
        loftr_matcher = self._get_matcher('loftr')
        superglue_matcher = self._get_matcher('superglue')
        
        matches = {}
        iterator = progress_bar(index_pairs) if verbose else index_pairs
        
        for idx1, idx2 in iterator:
            fname1 = img_filenames[idx1]
            fname2 = img_filenames[idx2]
            
            try:
                # Get matches from different methods
                mkpts0_loftr, mkpts1_loftr, conf_loftr = loftr_matcher.match_pair(
                    fname1, fname2, self.config['loftr_input_size']
                )
                
                # Also match in reverse direction for LoFTR
                mkpts1_loftr_lr, mkpts0_loftr_lr, conf_loftr_lr = loftr_matcher.match_pair(
                    fname2, fname1, self.config['loftr_input_size']
                )
                
                # SuperGlue at multiple scales
                all_sg_mkpts0, all_sg_mkpts1, all_sg_conf = [], [], []
                
                for size in self.config['superglue_sizes']:
                    mkpts0_sg, mkpts1_sg, conf_sg = superglue_matcher.match_pair(
                        fname1, fname2, size,
                        tta_groups=[('orig', 'orig'), ('flip_lr', 'flip_lr')]
                    )
                    all_sg_mkpts0.append(mkpts0_sg)
                    all_sg_mkpts1.append(mkpts1_sg)
                    all_sg_conf.append(conf_sg)
                
                # Combine all matches
                mkpts0 = np.concatenate([
                    mkpts0_loftr, mkpts0_loftr_lr
                ] + all_sg_mkpts0, axis=0)
                
                mkpts1 = np.concatenate([
                    mkpts1_loftr, mkpts1_loftr_lr
                ] + all_sg_mkpts1, axis=0)
                
                confidence = np.concatenate([
                    conf_loftr, conf_loftr_lr
                ] + all_sg_conf, axis=0)
                
                # Save if enough matches
                if len(mkpts0) >= self.config['min_matches']:
                    matches[(idx1, idx2)] = {
                        'keypoints1': mkpts0,
                        'keypoints2': mkpts1,
                        'confidence': confidence,
                        'num_matches': len(mkpts0)
                    }
                    logger.debug(f"Matched {fname1} - {fname2}: {len(mkpts0)} matches")
                else:
                    logger.warning(f"Too few matches for {fname1} - {fname2}: {len(mkpts0)}")
                    
            except Exception as e:
                logger.error(f"Error matching {fname1} - {fname2}: {e}")
                continue
        
        return matches
    
    def _match_superglue_only(
        self,
        img_filenames: List[str],
        index_pairs: List[Tuple[int, int]],
        verbose: bool
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Match using only SuperGlue with high keypoint count.
        """
        logger.info(f"Matching {len(index_pairs)} pairs using SuperGlue")
        
        # Use high-resolution SuperGlue
        superglue_matcher = self._get_matcher('superglue_high_res')
        
        matches = {}
        iterator = progress_bar(index_pairs) if verbose else index_pairs
        
        for idx1, idx2 in iterator:
            fname1 = img_filenames[idx1]
            fname2 = img_filenames[idx2]
            
            try:
                # Get matches at multiple scales
                all_mkpts0, all_mkpts1, all_conf = [], [], []
                
                for size in self.config['superglue_sizes']:
                    mkpts0, mkpts1, conf = superglue_matcher.match_pair(
                        fname1, fname2, size
                    )
                    all_mkpts0.append(mkpts0)
                    all_mkpts1.append(mkpts1)
                    all_conf.append(conf)
                
                # Combine matches
                mkpts0 = np.concatenate(all_mkpts0, axis=0)
                mkpts1 = np.concatenate(all_mkpts1, axis=0)
                confidence = np.concatenate(all_conf, axis=0)
                
                # Save if enough matches
                if len(mkpts0) >= self.config['min_matches']:
                    matches[(idx1, idx2)] = {
                        'keypoints1': mkpts0,
                        'keypoints2': mkpts1,
                        'confidence': confidence,
                        'num_matches': len(mkpts0)
                    }
                    logger.debug(f"Matched {fname1} - {fname2}: {len(mkpts0)} matches")
                else:
                    logger.warning(f"Too few matches for {fname1} - {fname2}: {len(mkpts0)}")
                    
            except Exception as e:
                logger.error(f"Error matching {fname1} - {fname2}: {e}")
                continue
        
        return matches
    
    def _match_loftr_only(
        self,
        img_filenames: List[str],
        index_pairs: List[Tuple[int, int]],
        verbose: bool
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Match using only LoFTR.
        """
        logger.info(f"Matching {len(index_pairs)} pairs using LoFTR")
        
        loftr_matcher = self._get_matcher('loftr')
        
        matches = {}
        iterator = progress_bar(index_pairs) if verbose else index_pairs
        
        for idx1, idx2 in iterator:
            fname1 = img_filenames[idx1]
            fname2 = img_filenames[idx2]
            
            try:
                # Match at multiple scales
                mkpts0, mkpts1, confidence = loftr_matcher.match_multi_scale(
                    fname1, fname2,
                    scales=[640, 1024, 1440]
                )
                
                # Save if enough matches
                if len(mkpts0) >= self.config['min_matches']:
                    matches[(idx1, idx2)] = {
                        'keypoints1': mkpts0,
                        'keypoints2': mkpts1,
                        'confidence': confidence,
                        'num_matches': len(mkpts0)
                    }
                    logger.debug(f"Matched {fname1} - {fname2}: {len(mkpts0)} matches")
                else:
                    logger.warning(f"Too few matches for {fname1} - {fname2}: {len(mkpts0)}")
                    
            except Exception as e:
                logger.error(f"Error matching {fname1} - {fname2}: {e}")
                continue
        
        return matches
    
    def _save_matches(
        self,
        matches: Dict[Tuple[int, int], Dict],
        img_filenames: List[str],
        output_dir: str
    ):
        """Save matches to h5 files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw matches
        raw_matches_path = os.path.join(output_dir, 'matches_loftr.h5')
        with h5py.File(raw_matches_path, mode='w') as f:
            for (idx1, idx2), match_data in matches.items():
                key1 = os.path.basename(img_filenames[idx1])
                key2 = os.path.basename(img_filenames[idx2])
                
                # Create group for key1 if it doesn't exist
                if key1 not in f:
                    f.create_group(key1)
                
                # Save concatenated keypoints
                concat_kpts = np.concatenate([
                    match_data['keypoints1'],
                    match_data['keypoints2']
                ], axis=1)
                
                f[key1].create_dataset(key2, data=concat_kpts)
        
        logger.info(f"Saved raw matches to {raw_matches_path}")
        
        # Process to unique keypoints
        process_matches_to_unique_keypoints(output_dir)
    
    def extract_and_match_features(
        self,
        image_paths: List[str],
        pairs: List[Tuple[int, int]],
        output_dir: str = './features',
        method: str = 'auto',
        verbose: bool = True
    ) -> str:
        """
        Complete feature extraction and matching pipeline.
        
        Args:
            image_paths: List of image file paths
            pairs: List of image pairs to match
            output_dir: Output directory for features
            method: Matching method
            verbose: Show progress
            
        Returns:
            Path to the feature directory
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run matching
        matches = self.match_images(
            image_paths,
            pairs,
            method=method,
            output_dir=output_dir,
            verbose=verbose
        )
        
        # Print summary
        logger.info(f"Feature extraction complete:")
        logger.info(f"  - Processed {len(pairs)} image pairs")
        logger.info(f"  - Found matches for {len(matches)} pairs")
        logger.info(f"  - Features saved to: {output_path}")
        
        return str(output_path)