"""
Main pipeline for 3D reconstruction from images.

This module provides the core Vision3D pipeline that orchestrates:
1. Image pair selection using global descriptors
2. Feature detection and matching (LoFTR/SuperGlue)
3. COLMAP integration for 3D reconstruction
4. Result optimization and refinement
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from ..models import LoFTRMatcher, SuperGlueMatcher
from ..utils.image_pairs import ImagePairSelector
from ..utils.colmap_interface import ColmapInterface
from .feature_extraction import FeatureExtractor
from .reconstruction import ReconstructionEngine

logger = logging.getLogger(__name__)


class Vision3DPipeline:
    """
    Main orchestrator for 3D reconstruction from images.
    
    This class implements a complete pipeline that transforms a collection of 2D photos
    into a 3D model. It intelligently combines modern deep learning approaches 
    (LoFTR, SuperGlue) with classical geometric methods (COLMAP).
    
    Think of this class as the "conductor" of an orchestra, coordinating different
    components to work together harmoniously.
    
    Attributes:
        device: PyTorch device for GPU acceleration (cuda/cpu)
        matcher_type: Type of matcher to use:
            - 'loftr': Dense matching, best for textureless scenes
            - 'superglue': Sparse matching, faster and good for textured scenes
            - 'hybrid': Automatically chooses based on scene characteristics
        config: Configuration dictionary controlling pipeline behavior
    
    Example:
        >>> # Create pipeline with hybrid matching (recommended)
        >>> pipeline = Vision3DPipeline(matcher_type='hybrid')
        >>> 
        >>> # Provide list of image paths
        >>> images = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']
        >>> 
        >>> # Run reconstruction - this is where the magic happens!
        >>> reconstruction = pipeline.reconstruct(images)
        >>> 
        >>> # Export results in various formats
        >>> pipeline.export_results(reconstruction, 'output/')
    """
    
    def __init__(
        self,
        matcher_type: str = 'hybrid',
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the Vision3D pipeline.
        
        Args:
            matcher_type: Type of feature matcher ('loftr', 'superglue', 'hybrid')
            device: PyTorch device (defaults to CUDA if available)
            config: Custom configuration dictionary
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.matcher_type = matcher_type
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Initialize components
        self._init_components()
        logger.info(f"Vision3D Pipeline initialized with {matcher_type} matcher on {self.device}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for the pipeline."""
        return {
            'image_resize': 1024,
            'pair_selection': {
                'min_pairs': 20,
                'similarity_threshold': 0.6,
                'exhaustive_if_less': 20
            },
            'matching': {
                'min_matches': 15,
                'confidence_threshold': 0.2,
                'use_tta': True,
                'tta_variants': ['orig', 'flip_lr']
            },
            'reconstruction': {
                'min_model_size': 3,
                'ba_refine_focal': True,
                'ba_refine_principal': True,
                'ba_refine_distortion': True
            }
        }
    
    def _init_components(self):
        """Initialize pipeline components."""
        # Image pair selector
        self.pair_selector = ImagePairSelector(
            device=self.device,
            config=self.config['pair_selection']
        )
        
        # Feature matchers
        if self.matcher_type in ['loftr', 'hybrid']:
            self.loftr_matcher = LoFTRMatcher(
                device=self.device,
                config=self.config['matching']
            )
        
        if self.matcher_type in ['superglue', 'hybrid']:
            self.superglue_matcher = SuperGlueMatcher(
                device=self.device,
                config=self.config['matching']
            )
        
        # COLMAP interface
        self.colmap = ColmapInterface(config=self.config['reconstruction'])
        
        # Reconstruction engine
        self.reconstruction_engine = ReconstructionEngine(
            device=self.device,
            config=self.config['reconstruction']
        )
    
    def reconstruct(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Perform 3D reconstruction from a collection of images.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save intermediate results
            verbose: Whether to show progress bars
        
        Returns:
            Dictionary containing reconstruction results:
                - 'points3D': 3D point cloud
                - 'cameras': Camera parameters
                - 'images': Registered images with poses
                - 'statistics': Reconstruction statistics
        """
        logger.info(f"Starting reconstruction with {len(image_paths)} images")
        
        if output_dir is None:
            output_dir = Path('./vision3d_output')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Select image pairs
        logger.info("Step 1/4: Selecting image pairs...")
        image_pairs = self.pair_selector.select_pairs(
            image_paths,
            verbose=verbose
        )
        logger.info(f"Selected {len(image_pairs)} image pairs for matching")
        
        # Step 2: Extract features and match
        logger.info("Step 2/4: Extracting features and matching...")
        matches = self._extract_and_match(
            image_paths,
            image_pairs,
            output_dir / 'features',
            verbose=verbose
        )
        
        # Step 3: Import to COLMAP
        logger.info("Step 3/4: Importing to COLMAP database...")
        database_path = output_dir / 'colmap.db'
        self.colmap.import_features(
            image_paths,
            matches,
            database_path
        )
        
        # Step 4: Run reconstruction
        logger.info("Step 4/4: Running 3D reconstruction...")
        reconstruction = self.reconstruction_engine.reconstruct(
            database_path,
            image_paths[0].parent,  # Assume all images in same directory
            output_dir / 'sparse',
            verbose=verbose
        )
        
        # Add statistics
        reconstruction['statistics'] = self._compute_statistics(reconstruction)
        
        logger.info("Reconstruction completed successfully!")
        logger.info(f"Registered {len(reconstruction['images'])} / {len(image_paths)} images")
        logger.info(f"Reconstructed {len(reconstruction['points3D'])} 3D points")
        
        return reconstruction
    
    def _extract_and_match(
        self,
        image_paths: List[str],
        image_pairs: List[Tuple[int, int]],
        feature_dir: Path,
        verbose: bool = True
    ) -> Dict:
        """Extract features and perform matching."""
        feature_dir.mkdir(parents=True, exist_ok=True)
        
        all_matches = {}
        
        if self.matcher_type == 'hybrid' and len(image_pairs) >= 400:
            # Use SuperGlue for larger scenes
            logger.info("Using SuperGlue for large-scale matching")
            matches = self.superglue_matcher.match_pairs(
                image_paths,
                image_pairs,
                feature_dir,
                verbose=verbose
            )
            all_matches.update(matches)
        else:
            # Use LoFTR for smaller scenes or as specified
            logger.info(f"Using {self.matcher_type} matcher")
            if self.matcher_type in ['loftr', 'hybrid']:
                matches = self.loftr_matcher.match_pairs(
                    image_paths,
                    image_pairs,
                    feature_dir,
                    verbose=verbose
                )
                all_matches.update(matches)
            
            if self.matcher_type == 'superglue':
                matches = self.superglue_matcher.match_pairs(
                    image_paths,
                    image_pairs,
                    feature_dir,
                    verbose=verbose
                )
                all_matches.update(matches)
        
        return all_matches
    
    def _compute_statistics(self, reconstruction: Dict) -> Dict:
        """Compute reconstruction statistics."""
        stats = {
            'num_images': len(reconstruction['images']),
            'num_points3D': len(reconstruction['points3D']),
            'num_cameras': len(reconstruction['cameras']),
            'mean_reprojection_error': 0.0,
            'mean_track_length': 0.0,
            'point_cloud_bounds': None
        }
        
        if reconstruction['points3D']:
            points = np.array([p['xyz'] for p in reconstruction['points3D'].values()])
            stats['point_cloud_bounds'] = {
                'min': points.min(axis=0).tolist(),
                'max': points.max(axis=0).tolist(),
                'center': points.mean(axis=0).tolist()
            }
            
            errors = [p['error'] for p in reconstruction['points3D'].values()]
            stats['mean_reprojection_error'] = np.mean(errors)
            
            track_lengths = [len(p['image_ids']) for p in reconstruction['points3D'].values()]
            stats['mean_track_length'] = np.mean(track_lengths)
        
        return stats
    
    def export_results(
        self,
        reconstruction: Dict,
        output_path: str,
        formats: List[str] = ['ply', 'json', 'colmap']
    ):
        """
        Export reconstruction results in various formats.
        
        Args:
            reconstruction: Reconstruction dictionary from reconstruct()
            output_path: Base path for output files
            formats: List of export formats ('ply', 'json', 'colmap', 'obj')
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            if fmt == 'ply':
                self._export_ply(reconstruction, output_path / 'point_cloud.ply')
            elif fmt == 'json':
                self._export_json(reconstruction, output_path / 'reconstruction.json')
            elif fmt == 'colmap':
                self._export_colmap(reconstruction, output_path / 'colmap')
            elif fmt == 'obj':
                self._export_obj(reconstruction, output_path / 'model.obj')
            else:
                logger.warning(f"Unknown export format: {fmt}")
    
    def _export_ply(self, reconstruction: Dict, output_file: Path):
        """Export point cloud as PLY file."""
        # Implementation would go here
        pass
    
    def _export_json(self, reconstruction: Dict, output_file: Path):
        """Export reconstruction as JSON."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'cameras': reconstruction['cameras'],
            'images': reconstruction['images'],
            'statistics': reconstruction['statistics']
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _export_colmap(self, reconstruction: Dict, output_dir: Path):
        """Export in COLMAP format."""
        # Implementation would go here
        pass
    
    def _export_obj(self, reconstruction: Dict, output_file: Path):
        """Export as OBJ mesh file."""
        # Implementation would go here
        pass