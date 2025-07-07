"""
Main Vision3D pipeline orchestrator.

This module provides the high-level API for 3D reconstruction,
coordinating all the components and providing a simple interface
for users.
"""

import os
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time

from .feature_extraction import FeatureExtractor
from .reconstruction import ReconstructionEngine
from ..utils.image_pairs import ImagePairSelector, get_image_pairs_shortlist

logger = logging.getLogger(__name__)


class Vision3DPipeline:
    """
    Main pipeline for 3D reconstruction from images.
    
    This class provides a high-level API that coordinates:
    - Image pair selection
    - Feature extraction and matching
    - 3D reconstruction
    - Result export
    
    Example:
        >>> pipeline = Vision3DPipeline()
        >>> reconstruction = pipeline.reconstruct("path/to/images")
    """
    
    def __init__(self, config: Optional[Dict] = None, 
                 matcher_type: str = 'hybrid',
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the Vision3D pipeline.
        
        Args:
            config: Configuration dictionary
            matcher_type: Type of matcher ('loftr', 'superglue', 'hybrid')
            device: Device for computation ('cuda', 'cpu', or torch.device)
        """
        self.config = config or self._get_default_config()
        self.matcher_type = matcher_type
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        logger.info(f"Initialized Vision3D pipeline on device: {self.device}")
        
        # Initialize components
        self._pair_selector = None
        self._feature_extractor = None
        self._reconstruction_engine = None
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            'image_selection': {
                'min_pairs_per_image': 35,
                'similarity_threshold': 0.5,
                'exhaustive_if_less_than': 20,
                'model_name': 'tf_efficientnet_b7',
                'model_path': './efficientnet/tf_efficientnet_b7.pth'
            },
            'feature_extraction': {
                'min_matches': 15,
                'large_dataset_threshold': 400,
                'loftr_input_size': 1024,
                'superglue_sizes': [1024, 1440],
                'high_res_keypoints': 8192
            },
            'reconstruction': {
                'camera_model': 'simple-radial',
                'single_camera': False,
                'min_num_matches': 15,
                'exhaustive_matching': True,
                'export_format': 'ply'
            }
        }
    
    @property
    def pair_selector(self) -> ImagePairSelector:
        """Get or create pair selector."""
        if self._pair_selector is None:
            self._pair_selector = ImagePairSelector(
                device=self.device,
                config=self.config['image_selection']
            )
        return self._pair_selector
    
    @property
    def feature_extractor(self) -> FeatureExtractor:
        """Get or create feature extractor."""
        if self._feature_extractor is None:
            self._feature_extractor = FeatureExtractor(
                config=self.config['feature_extraction'],
                device=self.device
            )
        return self._feature_extractor
    
    @property
    def reconstruction_engine(self) -> ReconstructionEngine:
        """Get or create reconstruction engine."""
        if self._reconstruction_engine is None:
            # Prepare reconstruction config
            recon_config = {
                'colmap': self.config['reconstruction'],
                'min_num_matches': self.config['reconstruction'].get('min_num_matches', 15),
                'exhaustive_matching': self.config['reconstruction'].get('exhaustive_matching', True),
                'export_format': self.config['reconstruction'].get('export_format', 'ply')
            }
            self._reconstruction_engine = ReconstructionEngine(config=recon_config)
        return self._reconstruction_engine
    
    def reconstruct(
        self,
        image_path: Union[str, List[str]],
        output_dir: str = './output',
        image_pairs: Optional[List[tuple]] = None,
        skip_pair_selection: bool = False,
        skip_feature_extraction: bool = False,
        skip_reconstruction: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete 3D reconstruction pipeline.
        
        Args:
            image_path: Path to image directory or list of image paths
            output_dir: Output directory for results
            image_pairs: Optional pre-computed image pairs
            skip_pair_selection: Skip automatic pair selection
            skip_feature_extraction: Skip feature extraction
            skip_reconstruction: Skip 3D reconstruction
            verbose: Show progress information
            
        Returns:
            Dictionary containing reconstruction results
        """
        start_time = time.time()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get image list
        if isinstance(image_path, str):
            image_paths = self._get_image_paths(image_path)
        else:
            image_paths = image_path
        
        logger.info(f"Found {len(image_paths)} images")
        
        results = {
            'num_images': len(image_paths),
            'image_paths': image_paths,
            'output_dir': str(output_path)
        }
        
        # Step 1: Image pair selection
        if not skip_pair_selection and image_pairs is None:
            logger.info("Step 1/3: Selecting image pairs...")
            image_pairs = self._select_image_pairs(image_paths, verbose)
            results['num_pairs'] = len(image_pairs)
            logger.info(f"Selected {len(image_pairs)} image pairs")
        elif image_pairs is not None:
            results['num_pairs'] = len(image_pairs)
            logger.info(f"Using provided {len(image_pairs)} image pairs")
        
        # Step 2: Feature extraction and matching
        feature_dir = output_path / 'features'
        if not skip_feature_extraction:
            logger.info("Step 2/3: Extracting and matching features...")
            feature_path = self._extract_features(
                image_paths, 
                image_pairs,
                feature_dir,
                verbose
            )
            results['feature_path'] = feature_path
        else:
            logger.info("Skipping feature extraction")
            results['feature_path'] = str(feature_dir)
        
        # Step 3: 3D reconstruction
        if not skip_reconstruction:
            logger.info("Step 3/3: Running 3D reconstruction...")
            reconstruction_results = self._run_reconstruction(
                image_paths,
                feature_dir,
                output_path
            )
            results.update(reconstruction_results)
        else:
            logger.info("Skipping 3D reconstruction")
        
        # Calculate total time
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        
        return results
    
    def _get_image_paths(self, image_dir: str) -> List[str]:
        """Get list of image paths from directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for filename in sorted(os.listdir(image_dir)):
            if not filename.startswith('.') and any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_dir, filename))
        
        return image_paths
    
    def _select_image_pairs(
        self,
        image_paths: List[str],
        verbose: bool
    ) -> List[tuple]:
        """Select image pairs for matching."""
        # Determine method based on matcher type
        if self.matcher_type == 'hybrid':
            method = 'auto'
        else:
            method = self.matcher_type
        
        # Use shortlist function directly for efficiency
        pairs = get_image_pairs_shortlist(
            image_paths,
            similarity_threshold=self.config['image_selection']['similarity_threshold'],
            min_pairs_per_image=self.config['image_selection']['min_pairs_per_image'],
            exhaustive_if_less_than=self.config['image_selection']['exhaustive_if_less_than'],
            device=self.device,
            model_name=self.config['image_selection']['model_name'],
            model_path=self.config['image_selection']['model_path']
        )
        
        return pairs
    
    def _extract_features(
        self,
        image_paths: List[str],
        image_pairs: List[tuple],
        output_dir: Path,
        verbose: bool
    ) -> str:
        """Extract and match features."""
        # Determine matching method
        if self.matcher_type == 'hybrid':
            method = 'auto'
        elif self.matcher_type == 'loftr':
            method = 'loftr'
        elif self.matcher_type == 'superglue':
            method = 'superglue'
        else:
            method = 'auto'
        
        # Run feature extraction
        feature_path = self.feature_extractor.extract_and_match_features(
            image_paths=image_paths,
            pairs=image_pairs,
            output_dir=str(output_dir),
            method=method,
            verbose=verbose
        )
        
        return feature_path
    
    def _run_reconstruction(
        self,
        image_paths: List[str],
        feature_dir: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Run 3D reconstruction."""
        # Get image directory
        if image_paths:
            image_dir = os.path.dirname(image_paths[0])
        else:
            image_dir = '.'
        
        # Run reconstruction
        results = self.reconstruction_engine.reconstruct_from_features(
            image_dir=image_dir,
            feature_dir=str(feature_dir),
            output_dir=str(output_dir),
            database_name='colmap.db',
            run_reconstruction=True,
            export_results=True
        )
        
        return results
    
    def export_results(
        self,
        reconstruction: Any,
        output_path: str,
        formats: List[str] = None
    ):
        """
        Export reconstruction results in various formats.
        
        Args:
            reconstruction: Reconstruction object
            output_path: Output path
            formats: List of export formats ['ply', 'json', 'nvm']
        """
        formats = formats or ['ply', 'json']
        
        for format in formats:
            try:
                if format == 'ply':
                    self.reconstruction_engine.colmap.export_reconstruction(
                        reconstruction,
                        f"{output_path}.{format}",
                        format=format
                    )
                elif format == 'json':
                    # Export camera parameters and sparse points as JSON
                    self._export_json(reconstruction, f"{output_path}.json")
                else:
                    logger.warning(f"Export format {format} not supported")
            except Exception as e:
                logger.error(f"Failed to export {format}: {e}")
    
    def _export_json(self, reconstruction: Any, output_path: str):
        """Export reconstruction as JSON."""
        import json
        
        data = {
            'cameras': {},
            'images': {},
            'points3D': {}
        }
        
        # Export cameras
        for cam_id, camera in reconstruction.cameras.items():
            data['cameras'][str(cam_id)] = {
                'model': camera.model.name,
                'width': camera.width,
                'height': camera.height,
                'params': camera.params.tolist()
            }
        
        # Export images
        for img_id, image in reconstruction.images.items():
            data['images'][str(img_id)] = {
                'name': image.name,
                'camera_id': image.camera_id,
                'rotation': image.rotation_matrix().tolist(),
                'translation': image.translation.tolist()
            }
        
        # Export points (subsample for large reconstructions)
        max_points = 10000
        point_ids = list(reconstruction.points3D.keys())
        if len(point_ids) > max_points:
            import random
            point_ids = random.sample(point_ids, max_points)
        
        for point_id in point_ids:
            point = reconstruction.points3D[point_id]
            data['points3D'][str(point_id)] = {
                'xyz': point.xyz.tolist(),
                'rgb': point.color.tolist(),
                'error': float(point.error)
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported reconstruction to {output_path}")
    
    def visualize_matches(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None
    ):
        """
        Visualize matches between two images.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            output_path: Optional path to save visualization
        """
        # This would use the matcher's visualization method
        logger.info("Match visualization not yet implemented")
    
    @classmethod
    def from_config(cls, config_path: str) -> 'Vision3DPipeline':
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to configuration file (JSON/YAML)
            
        Returns:
            Vision3DPipeline instance
        """
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(config=config)