"""
3D reconstruction engine.

This module handles the actual 3D reconstruction process using COLMAP,
including database creation, feature import, and reconstruction execution.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..utils.colmap_interface import ColmapInterface

logger = logging.getLogger(__name__)


class ReconstructionEngine:
    """
    Manages 3D reconstruction using COLMAP.
    
    This class handles:
    - Creating COLMAP databases
    - Importing features and matches
    - Running incremental reconstruction
    - Exporting results in various formats
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the reconstruction engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.colmap = ColmapInterface(self.config['colmap'])
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'colmap': {
                'camera_model': 'simple-radial',
                'single_camera': False,
                'min_model_size': 3,
                'ba_refine_focal': True,
                'ba_refine_principal': True,
                'ba_refine_distortion': True,
                'ba_local_max_refinements': 2,
                'ba_global_max_refinements': 10
            },
            'reconstruction': {
                'min_num_matches': 15,
                'exhaustive_matching': True,
                'export_format': 'ply'
            }
        }
    
    def reconstruct_from_features(
        self,
        image_dir: str,
        feature_dir: str,
        output_dir: str,
        database_name: str = 'colmap.db',
        run_reconstruction: bool = True,
        export_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run 3D reconstruction from pre-computed features.
        
        Args:
            image_dir: Directory containing images
            feature_dir: Directory containing features (h5 files)
            output_dir: Output directory for reconstruction
            database_name: Name of COLMAP database
            run_reconstruction: Whether to run reconstruction after import
            export_results: Whether to export reconstruction results
            
        Returns:
            Dictionary with reconstruction results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Database path
        database_path = output_path / database_name
        
        # Remove existing database
        if database_path.exists():
            os.remove(database_path)
            logger.info(f"Removed existing database: {database_path}")
        
        # Import features to database
        logger.info("Importing features to COLMAP database...")
        fname_to_id = self.colmap.import_features_from_h5(
            image_dir=image_dir,
            feature_dir=feature_dir,
            database_path=str(database_path),
            camera_model=self.config['colmap']['camera_model'],
            single_camera=self.config['colmap']['single_camera']
        )
        
        logger.info(f"Imported {len(fname_to_id)} images to database")
        
        results = {
            'database_path': str(database_path),
            'num_images': len(fname_to_id),
            'image_id_map': fname_to_id
        }
        
        # Run reconstruction if requested
        if run_reconstruction:
            reconstruction_path = output_path / 'colmap_reconstruction'
            
            logger.info("Running COLMAP reconstruction...")
            maps = self.colmap.run_reconstruction(
                database_path=database_path,
                image_path=Path(image_dir),
                output_path=reconstruction_path,
                min_num_matches=self.config['reconstruction']['min_num_matches'],
                exhaustive_matching=self.config['reconstruction']['exhaustive_matching']
            )
            
            results['reconstructions'] = maps
            results['reconstruction_path'] = str(reconstruction_path)
            
            # Export results if requested
            if export_results and maps:
                self._export_reconstructions(
                    maps,
                    reconstruction_path,
                    self.config['reconstruction']['export_format']
                )
        
        return results
    
    def reconstruct_from_matches(
        self,
        image_paths: List[str],
        matches: Dict[Tuple[int, int], Dict],
        output_dir: str,
        database_name: str = 'colmap.db',
        run_reconstruction: bool = True,
        export_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run 3D reconstruction directly from matches.
        
        Args:
            image_paths: List of image file paths
            matches: Dictionary of matches
            output_dir: Output directory for reconstruction
            database_name: Name of COLMAP database
            run_reconstruction: Whether to run reconstruction after import
            export_results: Whether to export reconstruction results
            
        Returns:
            Dictionary with reconstruction results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Database path
        database_path = output_path / database_name
        
        # Remove existing database
        if database_path.exists():
            os.remove(database_path)
            logger.info(f"Removed existing database: {database_path}")
        
        # Import features to database
        logger.info("Importing features to COLMAP database...")
        self.colmap.import_features(
            image_paths=image_paths,
            matches=matches,
            database_path=database_path
        )
        
        logger.info(f"Imported {len(image_paths)} images and {len(matches)} matches")
        
        results = {
            'database_path': str(database_path),
            'num_images': len(image_paths),
            'num_matches': len(matches)
        }
        
        # Run reconstruction if requested
        if run_reconstruction:
            reconstruction_path = output_path / 'colmap_reconstruction'
            image_dir = os.path.dirname(image_paths[0]) if image_paths else '.'
            
            logger.info("Running COLMAP reconstruction...")
            maps = self.colmap.run_reconstruction(
                database_path=database_path,
                image_path=Path(image_dir),
                output_path=reconstruction_path,
                min_num_matches=self.config['reconstruction']['min_num_matches'],
                exhaustive_matching=self.config['reconstruction']['exhaustive_matching']
            )
            
            results['reconstructions'] = maps
            results['reconstruction_path'] = str(reconstruction_path)
            
            # Export results if requested
            if export_results and maps:
                self._export_reconstructions(
                    maps,
                    reconstruction_path,
                    self.config['reconstruction']['export_format']
                )
        
        return results
    
    def _export_reconstructions(
        self,
        reconstructions: Dict[int, Any],
        output_path: Path,
        format: str = 'ply'
    ):
        """Export reconstructions to various formats."""
        for idx, reconstruction in reconstructions.items():
            # Export model
            export_path = output_path / f'model_{idx}'
            export_path.mkdir(exist_ok=True)
            
            self.colmap.export_reconstruction(
                reconstruction,
                str(export_path / f'model.{format}'),
                format=format
            )
            
            # Save statistics
            stats_path = export_path / 'stats.txt'
            self._save_reconstruction_stats(reconstruction, stats_path)
            
            logger.info(f"Exported model {idx} to {export_path}")
    
    def _save_reconstruction_stats(self, reconstruction: Any, output_path: Path):
        """Save reconstruction statistics."""
        num_images = len(reconstruction.images)
        num_points = len(reconstruction.points3D)
        
        # Calculate additional statistics
        if num_points > 0:
            # Mean reprojection error
            errors = []
            for point_id, point in reconstruction.points3D.items():
                errors.append(point.error)
            mean_error = sum(errors) / len(errors) if errors else 0
            
            # Point cloud bounds
            points = np.array([point.xyz for point in reconstruction.points3D.values()])
            min_bounds = points.min(axis=0)
            max_bounds = points.max(axis=0)
        else:
            mean_error = 0
            min_bounds = max_bounds = [0, 0, 0]
        
        # Write statistics
        with open(output_path, 'w') as f:
            f.write("Reconstruction Statistics\n")
            f.write("========================\n\n")
            f.write(f"Number of images: {num_images}\n")
            f.write(f"Number of 3D points: {num_points}\n")
            f.write(f"Mean reprojection error: {mean_error:.4f}\n")
            f.write(f"\nPoint cloud bounds:\n")
            f.write(f"  X: [{min_bounds[0]:.3f}, {max_bounds[0]:.3f}]\n")
            f.write(f"  Y: [{min_bounds[1]:.3f}, {max_bounds[1]:.3f}]\n")
            f.write(f"  Z: [{min_bounds[2]:.3f}, {max_bounds[2]:.3f}]\n")
            
            # Camera statistics
            if num_images > 0:
                f.write(f"\nCamera statistics:\n")
                for img_id, image in reconstruction.images.items():
                    camera = reconstruction.cameras[image.camera_id]
                    f.write(f"  Image {image.name}:\n")
                    f.write(f"    Camera model: {camera.model_name}\n")
                    f.write(f"    Focal length: {camera.focal_length:.2f}\n")
    
    def visualize_reconstruction(
        self,
        reconstruction_path: str,
        output_path: Optional[str] = None
    ):
        """
        Visualize reconstruction results.
        
        Args:
            reconstruction_path: Path to reconstruction
            output_path: Optional path to save visualization
        """
        # This would use visualization tools to create
        # interactive 3D visualizations of the reconstruction
        logger.info("Visualization not yet implemented")
    
    def evaluate_reconstruction(
        self,
        reconstruction: Any,
        ground_truth_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Args:
            reconstruction: COLMAP reconstruction object
            ground_truth_path: Optional path to ground truth
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['num_images'] = len(reconstruction.images)
        metrics['num_points'] = len(reconstruction.points3D)
        
        # Reprojection error
        if metrics['num_points'] > 0:
            errors = [point.error for point in reconstruction.points3D.values()]
            metrics['mean_reprojection_error'] = sum(errors) / len(errors)
            metrics['median_reprojection_error'] = sorted(errors)[len(errors) // 2]
        
        # Track length statistics
        if metrics['num_points'] > 0:
            track_lengths = [len(point.image_ids) for point in reconstruction.points3D.values()]
            metrics['mean_track_length'] = sum(track_lengths) / len(track_lengths)
            metrics['max_track_length'] = max(track_lengths)
        
        # If ground truth is provided, compute additional metrics
        if ground_truth_path:
            # This would load ground truth and compute metrics like
            # absolute trajectory error, relative pose error, etc.
            logger.info("Ground truth evaluation not yet implemented")
        
        return metrics


# Import numpy at module level for statistics
import numpy as np