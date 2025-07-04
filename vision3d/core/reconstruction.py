"""
3D reconstruction engine using COLMAP.

This module handles the actual 3D reconstruction process,
including incremental mapping, bundle adjustment, and optimization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


class ReconstructionEngine:
    """
    Engine for 3D reconstruction using COLMAP.
    
    This class manages:
    - Incremental reconstruction
    - Bundle adjustment
    - Multi-model merging
    - Quality assessment
    """
    
    def __init__(self, device, config: Optional[Dict] = None):
        """
        Initialize reconstruction engine.
        
        Args:
            device: PyTorch device (for GPU-accelerated operations)
            config: Configuration dictionary
        """
        self.device = device
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default reconstruction configuration."""
        return {
            'min_model_size': 3,  # IMC-2023 setting for smaller reconstructions
            'max_reproj_error': 4.0,
            'min_triangulation_angle': 1.5,
            'ba_refine_focal': True,
            'ba_refine_principal': True,
            'ba_refine_distortion': True,
            'ba_local_max_refinements': 2,  # IMC-2023 setting
            'ba_global_max_refinements': 20,  # IMC-2023 setting
            'filter_max_reproj_error': 4.0,
            'filter_min_tri_angle': 1.5
        }
    
    def reconstruct(
        self,
        database_path: Path,
        image_path: Path,
        output_path: Path,
        verbose: bool = True
    ) -> Dict:
        """
        Perform 3D reconstruction.
        
        Args:
            database_path: Path to COLMAP database
            image_path: Path to images directory
            output_path: Output directory for reconstruction
            verbose: Show progress
        
        Returns:
            Dictionary containing reconstruction results
        """
        import pycolmap
        
        logger.info("Starting 3D reconstruction...")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get mapper options
        mapper_options = self._get_mapper_options()
        
        # Run incremental mapping
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(output_path),
            options=mapper_options
        )
        
        # Process reconstructions
        if isinstance(reconstructions, dict) and reconstructions:
            # Multiple reconstructions found
            best_idx, best_reconstruction = self._select_best_reconstruction(reconstructions)
            reconstruction_data = self._process_reconstruction(best_reconstruction)
            
            # Merge other reconstructions if beneficial
            for idx, recon in reconstructions.items():
                if idx != best_idx:
                    self._try_merge_reconstruction(reconstruction_data, recon)
        else:
            # Single or no reconstruction
            logger.warning("No valid reconstruction found")
            reconstruction_data = self._create_empty_reconstruction()
        
        # Post-process and optimize
        if reconstruction_data['num_images'] > 0:
            reconstruction_data = self._post_process(reconstruction_data)
        
        logger.info(f"Reconstruction completed: {reconstruction_data['num_images']} images registered")
        
        return reconstruction_data
    
    def _get_mapper_options(self):
        """Get COLMAP mapper options."""
        import pycolmap
        
        options = pycolmap.IncrementalPipelineOptions()
        
        # Set mapper options
        mapper_options = options.mapper
        mapper_options.init_min_tri_angle = self.config['min_triangulation_angle']
        mapper_options.filter_max_reproj_error = self.config['filter_max_reproj_error']
        mapper_options.filter_min_tri_angle = self.config['filter_min_tri_angle']
        
        # Registration options - lowered thresholds for more permissive registration
        mapper_options.abs_pose_min_num_inliers = 8  # Lowered from 15
        mapper_options.abs_pose_min_inlier_ratio = 0.15  # Lowered from 0.25
        
        # IMC-2023 optimized parameters
        # min_model_size is on the pipeline options, not mapper
        options.min_model_size = self.config.get('min_model_size', 3)  # Allow smaller reconstructions
        
        # Bundle adjustment options are also on pipeline options
        options.ba_local_max_refinements = self.config.get('ba_local_max_refinements', 2)  # Faster local BA
        options.ba_global_max_refinements = self.config.get('ba_global_max_refinements', 20)  # More global optimization
        
        # Bundle adjustment refinement options
        mapper_options.abs_pose_refine_focal_length = self.config.get('ba_refine_focal', True)
        mapper_options.abs_pose_refine_extra_params = self.config.get('ba_refine_distortion', True)
        
        # Efficiency options for production
        mapper_options.num_threads = -1  # Use all available threads
        
        return options
    
    def _select_best_reconstruction(self, reconstructions: Dict) -> object:
        """Select the best reconstruction from multiple models."""
        best_idx = None
        best_score = -1
        best_recon = None
        
        for idx, recon in reconstructions.items():
            # Score based on number of registered images and 3D points
            score = len(recon.images) * 1000 + len(recon.points3D)
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_recon = recon
        
        logger.info(f"Selected reconstruction {best_idx} with score {best_score}")
        return best_idx, best_recon
    
    def _process_reconstruction(self, reconstruction) -> Dict:
        """Process COLMAP reconstruction into our format."""
        data = {
            'cameras': {},
            'images': {},
            'points3D': {},
            'num_cameras': len(reconstruction.cameras),
            'num_images': len(reconstruction.images),
            'num_points3D': len(reconstruction.points3D)
        }
        
        # Process cameras
        for cam_id, camera in reconstruction.cameras.items():
            # Handle different pycolmap API versions
            if hasattr(camera, 'model_name'):
                model_name = camera.model_name
            elif hasattr(camera, 'model'):
                # Map model ID to name
                model_names = {
                    0: 'SIMPLE_PINHOLE',
                    1: 'PINHOLE', 
                    2: 'SIMPLE_RADIAL',
                    3: 'RADIAL',
                    4: 'OPENCV'
                }
                model_name = model_names.get(camera.model, f'UNKNOWN_{camera.model}')
            else:
                model_name = 'UNKNOWN'
                
            data['cameras'][cam_id] = {
                'model': model_name,
                'width': camera.width,
                'height': camera.height,
                'params': camera.params.tolist(),
                'focal_length': self._get_focal_length(camera)
            }
        
        # Process images
        for img_id, image in reconstruction.images.items():
            # Extract rotation and translation from cam_from_world transform
            if hasattr(image, 'cam_from_world'):
                # New API: cam_from_world contains the transformation
                transform = image.cam_from_world
                # Extract rotation matrix and translation vector
                R = transform.rotation.matrix()
                t = transform.translation
            elif hasattr(image, 'rotmat'):
                # Old API
                R = image.rotmat()
                t = image.tvec
            else:
                # Fallback: try to extract from quaternion
                if hasattr(image, 'qvec'):
                    from scipy.spatial.transform import Rotation
                    R = Rotation.from_quat(image.qvec).as_matrix()
                    t = image.tvec if hasattr(image, 'tvec') else np.zeros(3)
                else:
                    R = np.eye(3)
                    t = np.zeros(3)
            
            data['images'][image.name] = {
                'image_id': img_id,
                'camera_id': image.camera_id,
                'R': R.tolist(),
                't': t.tolist(),
                'qvec': transform.rotation.quat if hasattr(image, 'cam_from_world') else getattr(image, 'qvec', [1, 0, 0, 0]),
                'num_points3D': image.num_points3D if hasattr(image, 'num_points3D') else 0
            }
        
        # Process 3D points
        for pt_id, point3D in reconstruction.points3D.items():
            # Extract track information (image observations)
            if hasattr(point3D, 'track'):
                # New API: track contains the observations
                track = point3D.track
                image_ids = []
                point2D_idxs = []
                for element in track.elements:
                    image_ids.append(element.image_id)
                    point2D_idxs.append(element.point2D_idx)
            elif hasattr(point3D, 'image_ids'):
                # Old API
                image_ids = point3D.image_ids.tolist()
                point2D_idxs = point3D.point2D_idxs.tolist()
            else:
                image_ids = []
                point2D_idxs = []
            
            data['points3D'][pt_id] = {
                'xyz': point3D.xyz.tolist(),
                'rgb': point3D.color.tolist() if hasattr(point3D, 'color') else [128, 128, 128],
                'error': point3D.error if hasattr(point3D, 'error') else 0.0,
                'image_ids': image_ids,
                'point2D_idxs': point2D_idxs
            }
        
        return data
    
    def _get_focal_length(self, camera) -> float:
        """Extract focal length from camera parameters."""
        # Get model ID if available
        if hasattr(camera, 'model'):
            model_id = camera.model
        else:
            # Try to determine from model_name if available
            if hasattr(camera, 'model_name'):
                model_map = {
                    'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4
                }
                model_id = model_map.get(camera.model_name, 0)
            else:
                model_id = 2  # Default to SIMPLE_RADIAL
        
        # Extract focal length based on model
        if model_id in [0, 2]:  # SIMPLE_PINHOLE, SIMPLE_RADIAL
            return camera.params[0]
        elif model_id in [1, 3, 4]:  # PINHOLE, RADIAL, OPENCV
            return (camera.params[0] + camera.params[1]) / 2
        else:
            return camera.params[0]
    
    def _try_merge_reconstruction(self, main_recon: Dict, other_recon):
        """Try to merge another reconstruction into the main one."""
        # Check if reconstructions share common images
        main_images = set(main_recon['images'].keys())
        other_images = set(img.name for img in other_recon.images.values())
        
        common_images = main_images.intersection(other_images)
        
        if len(common_images) >= 3:
            logger.info(f"Found {len(common_images)} common images, attempting merge...")
            # In production, this would implement proper reconstruction merging
            # using common images to compute relative transformation
    
    def _create_empty_reconstruction(self) -> Dict:
        """Create empty reconstruction structure."""
        return {
            'cameras': {},
            'images': {},
            'points3D': {},
            'num_cameras': 0,
            'num_images': 0,
            'num_points3D': 0
        }
    
    def _post_process(self, reconstruction: Dict) -> Dict:
        """Post-process reconstruction for quality improvement."""
        # Filter high-error points
        if self.config['filter_max_reproj_error'] > 0:
            reconstruction = self._filter_high_error_points(reconstruction)
        
        # Remove outlier points
        reconstruction = self._remove_outliers(reconstruction)
        
        # Optimize point cloud
        reconstruction = self._optimize_point_cloud(reconstruction)
        
        return reconstruction
    
    def _filter_high_error_points(self, reconstruction: Dict) -> Dict:
        """Filter points with high reprojection error."""
        filtered_points = {}
        threshold = self.config['filter_max_reproj_error']
        
        for pt_id, point in reconstruction['points3D'].items():
            if point['error'] <= threshold:
                filtered_points[pt_id] = point
        
        num_filtered = len(reconstruction['points3D']) - len(filtered_points)
        if num_filtered > 0:
            logger.info(f"Filtered {num_filtered} high-error points")
        
        reconstruction['points3D'] = filtered_points
        reconstruction['num_points3D'] = len(filtered_points)
        
        return reconstruction
    
    def _remove_outliers(self, reconstruction: Dict) -> Dict:
        """Remove statistical outliers from point cloud."""
        if not reconstruction['points3D']:
            return reconstruction
        
        # Extract point coordinates
        points = np.array([p['xyz'] for p in reconstruction['points3D'].values()])
        
        # Compute statistics
        centroid = points.mean(axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Remove points beyond 3 standard deviations
        mean_dist = distances.mean()
        std_dist = distances.std()
        threshold = mean_dist + 3 * std_dist
        
        # Filter points
        filtered_points = {}
        for i, (pt_id, point) in enumerate(reconstruction['points3D'].items()):
            if distances[i] <= threshold:
                filtered_points[pt_id] = point
        
        num_filtered = len(reconstruction['points3D']) - len(filtered_points)
        if num_filtered > 0:
            logger.info(f"Removed {num_filtered} outlier points")
        
        reconstruction['points3D'] = filtered_points
        reconstruction['num_points3D'] = len(filtered_points)
        
        return reconstruction
    
    def _optimize_point_cloud(self, reconstruction: Dict) -> Dict:
        """Optimize point cloud for better quality."""
        # This would implement various optimization strategies:
        # - Point cloud densification
        # - Color consistency optimization
        # - Geometric refinement
        
        return reconstruction
    
    def evaluate_reconstruction(self, reconstruction: Dict) -> Dict:
        """
        Evaluate reconstruction quality.
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'completeness': 0.0,
            'accuracy': 0.0,
            'coverage': 0.0,
            'mean_reproj_error': 0.0,
            'median_reproj_error': 0.0,
            'point_density': 0.0
        }
        
        if not reconstruction['points3D']:
            return metrics
        
        # Compute reprojection error statistics
        errors = [p['error'] for p in reconstruction['points3D'].values()]
        metrics['mean_reproj_error'] = np.mean(errors)
        metrics['median_reproj_error'] = np.median(errors)
        
        # Compute coverage (registered images / total images)
        if 'total_images' in reconstruction:
            metrics['coverage'] = reconstruction['num_images'] / reconstruction['total_images']
        
        # Compute point density
        points = np.array([p['xyz'] for p in reconstruction['points3D'].values()])
        if len(points) > 0:
            # Points per cubic unit
            bbox_min = points.min(axis=0)
            bbox_max = points.max(axis=0)
            volume = np.prod(bbox_max - bbox_min)
            metrics['point_density'] = len(points) / volume if volume > 0 else 0
        
        return metrics