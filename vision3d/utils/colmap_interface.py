"""
COLMAP interface for 3D reconstruction.

This module provides a clean interface to COLMAP functionality,
handling database creation, feature import, and reconstruction.
"""

import os
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import h5py
import pycolmap
from collections import defaultdict
from copy import deepcopy

logger = logging.getLogger(__name__)


# COLMAP Database utilities
def array_to_blob(array: np.ndarray) -> bytes:
    """Convert numpy array to blob for SQLite storage."""
    return array.tobytes()


def image_ids_to_pair_id(image_id1: int, image_id2: int, max_image_id: int = 2**31 - 1) -> int:
    """Convert pair of image IDs to unique pair ID."""
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * max_image_id + image_id2


def pair_id_to_image_ids(pair_id: int, max_image_id: int = 2**31 - 1) -> Tuple[int, int]:
    """Convert pair ID back to image IDs."""
    image_id2 = pair_id % max_image_id
    image_id1 = (pair_id - image_id2) // max_image_id
    return image_id1, image_id2


def get_focal_length(image_path: str, err_on_default: bool = False) -> float:
    """
    Extract focal length from image EXIF data.
    
    Args:
        image_path: Path to the image file
        err_on_default: If True, raise error when focal length not found
        
    Returns:
        Focal length in pixels
    """
    from PIL import Image, ExifTags
    
    image = Image.open(image_path)
    max_size = max(image.size)
    
    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
                break
        
        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size
    
    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")
        
        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
        logger.warning(f"No focal length found for {image_path}, using prior {FOCAL_PRIOR}")
    
    return focal


class COLMAPDatabase(sqlite3.Connection):
    """
    COLMAP database interface.
    
    Based on COLMAP's database.py script.
    """
    
    @staticmethod
    def connect(database_path: str):
        """Connect to COLMAP database."""
        return sqlite3.connect(database_path, factory=COLMAPDatabase)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables()
    
    def create_tables(self):
        """Create necessary COLMAP tables."""
        # Create cameras table
        self.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB,
                prior_focal_length INTEGER NOT NULL
            )
        """)
        
        # Create images table
        self.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            )
        """)
        
        # Create keypoints table
        self.execute("""
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            )
        """)
        
        # Create matches table
        self.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB
            )
        """)
        
        # Create descriptors table
        self.execute("""
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            )
        """)
        
        # Create two_view_geometries table
        self.execute("""
            CREATE TABLE IF NOT EXISTS two_view_geometries (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                config INTEGER NOT NULL,
                F BLOB,
                E BLOB,
                H BLOB
            )
        """)
        
        self.commit()
    
    def add_camera(
        self,
        model: int,
        width: int,
        height: int,
        params: np.ndarray,
        prior_focal_length: bool = False,
        camera_id: Optional[int] = None
    ) -> int:
        """Add camera to database."""
        params = np.asarray(params, np.float64)
        
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params), prior_focal_length)
        )
        return cursor.lastrowid
    
    def add_image(
        self,
        name: str,
        camera_id: int,
        prior_q: np.ndarray = None,
        prior_t: np.ndarray = None,
        image_id: Optional[int] = None
    ) -> int:
        """Add image to database."""
        prior_q = prior_q if prior_q is not None else np.array([1.0, 0.0, 0.0, 0.0])
        prior_t = prior_t if prior_t is not None else np.zeros(3)
        
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, 
             prior_q[0], prior_q[1], prior_q[2], prior_q[3],
             prior_t[0], prior_t[1], prior_t[2])
        )
        return cursor.lastrowid
    
    def add_keypoints(self, image_id: int, keypoints: np.ndarray):
        """Add keypoints to database."""
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]
        
        keypoints = np.asarray(keypoints, np.float32)
        
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),)
        )
    
    def add_descriptors(self, image_id: int, descriptors: np.ndarray):
        """Add descriptors to database."""
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),)
        )
    
    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray):
        """Add matches to database."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),)
        )
    
    def add_two_view_geometry(self, image_id1: int, image_id2: int,
                            matches: np.ndarray,
                            F: np.ndarray = None,
                            E: np.ndarray = None,
                            H: np.ndarray = None,
                            config: int = 2) -> None:
        """Add two-view geometry for an image pair."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        
        F = F if F is not None else np.eye(3)
        E = E if E is not None else np.eye(3)
        H = H if H is not None else np.eye(3)
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H))
        )


def process_matches_to_unique_keypoints(feature_dir: str) -> None:
    """
    Process raw matches to extract unique keypoints and clean match indices.
    
    This function reads the raw matches, identifies unique keypoints across
    all images, and creates cleaned match indices that reference these
    unique keypoints.
    """
    logger.info("Processing matches to extract unique keypoints")
    
    # Import function from image_pairs to avoid circular import
    from .image_pairs import get_unique_indices
    
    # Collect all keypoints and match indices
    keypoints = defaultdict(list)
    match_indices = defaultdict(dict)
    total_keypoints = defaultdict(int)
    
    # Read raw matches
    with h5py.File(f'{feature_dir}/matches_loftr.h5', mode='r') as f:
        for key1 in f.keys():
            group = f[key1]
            for key2 in group.keys():
                matches = group[key2][...]
                
                # Extract keypoints
                keypoints[key1].append(matches[:, :2])
                keypoints[key2].append(matches[:, 2:])
                
                # Create match indices
                import torch
                current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                current_match[:, 0] += total_keypoints[key1]
                current_match[:, 1] += total_keypoints[key2]
                
                total_keypoints[key1] += len(matches)
                total_keypoints[key2] += len(matches)
                match_indices[key1][key2] = current_match
    
    # Round keypoints for unique detection
    for k in keypoints.keys():
        keypoints[k] = np.round(np.concatenate(keypoints[k], axis=0))
    
    # Find unique keypoints
    unique_keypoints = {}
    unique_match_indices = {}
    
    for k in keypoints.keys():
        import torch
        unique_kpts, unique_reverse_indices = torch.unique(
            torch.from_numpy(keypoints[k]),
            dim=0,
            return_inverse=True
        )
        unique_match_indices[k] = unique_reverse_indices
        unique_keypoints[k] = unique_kpts.numpy()
    
    # Create cleaned matches
    cleaned_matches = defaultdict(dict)
    
    for k1, group in match_indices.items():
        for k2, matches in group.items():
            # Remap to unique indices
            remapped = deepcopy(matches)
            remapped[:, 0] = unique_match_indices[k1][remapped[:, 0]]
            remapped[:, 1] = unique_match_indices[k2][remapped[:, 1]]
            
            # Remove duplicates
            mkpts = np.concatenate([
                unique_keypoints[k1][remapped[:, 0]],
                unique_keypoints[k2][remapped[:, 1]]
            ], axis=1)
            
            unique_indices = get_unique_indices(torch.from_numpy(mkpts), dim=0)
            remapped = remapped[unique_indices]
            
            # Ensure one-to-one matching
            unique_indices1 = get_unique_indices(remapped[:, 0], dim=0)
            remapped = remapped[unique_indices1]
            
            unique_indices2 = get_unique_indices(remapped[:, 1], dim=0)
            remapped = remapped[unique_indices2]
            
            cleaned_matches[k1][k2] = remapped.numpy()
    
    # Save unique keypoints
    with h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f:
        for k, kpts in unique_keypoints.items():
            f[k] = kpts
    
    # Save cleaned matches
    with h5py.File(f'{feature_dir}/matches.h5', mode='w') as f:
        for k1, group in cleaned_matches.items():
            group_h5 = f.require_group(k1)
            for k2, matches in group.items():
                group_h5[k2] = matches
    
    logger.info("Finished processing matches")


class ColmapInterface:
    """
    Interface for COLMAP database and reconstruction operations.
    
    This class handles:
    - Creating and managing COLMAP databases
    - Importing features and matches
    - Running incremental reconstruction
    - Exporting results
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize COLMAP interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default COLMAP configuration."""
        return {
            'camera_model': 'simple-radial',
            'single_camera': False,
            'min_model_size': 3,
            'ba_refine_focal': True,
            'ba_refine_principal': True,
            'ba_refine_distortion': True,
            'ba_local_max_refinements': 2,
            'ba_global_max_refinements': 10
        }
    
    def import_features(
        self,
        image_paths: List[str],
        matches: Dict,
        database_path: Path
    ):
        """
        Import features and matches into COLMAP database.
        
        Args:
            image_paths: List of image paths
            matches: Dictionary of matches from matchers
            database_path: Path to COLMAP database
        """
        logger.info(f"Creating COLMAP database at {database_path}")
        
        # Create database
        db = COLMAPDatabase.connect(str(database_path))
        db.create_tables()
        
        # Import images and keypoints
        image_id_map = self._import_images_and_keypoints(
            db, image_paths, matches
        )
        
        # Import matches
        self._import_matches(db, matches, image_id_map)
        
        db.commit()
        db.close()
        
        logger.info("Features imported to COLMAP database")
    
    def _import_images_and_keypoints(
        self,
        db,
        image_paths: List[str],
        matches: Dict
    ) -> Dict[int, int]:
        """Import images and keypoints to database."""
        image_id_map = {}
        
        # First, collect all unique keypoints for each image
        image_keypoints = self._collect_image_keypoints(image_paths, matches)
        
        # Import each image
        for idx, img_path in enumerate(image_paths):
            # Create camera
            camera_id = self._create_camera(db, img_path)
            
            # Add image
            image_name = Path(img_path).name
            logger.debug(f"Adding image {idx}: {image_name}")
            image_id = db.add_image(image_name, camera_id)
            image_id_map[idx] = image_id
            
            # Add keypoints
            if idx in image_keypoints:
                keypoints = image_keypoints[idx]
                logger.debug(f"Image {idx} keypoints shape: {keypoints.shape}")
                # Only add keypoints if there are any
                if keypoints.shape[0] > 0 and keypoints.shape[1] > 0:
                    db.add_keypoints(image_id, keypoints)
                    # Add dummy descriptors (COLMAP requires them)
                    # Using 128-dimensional zero descriptors
                    descriptors = np.zeros((keypoints.shape[0], 128), dtype=np.uint8)
                    db.add_descriptors(image_id, descriptors)
        
        return image_id_map
    
    def _collect_image_keypoints(
        self,
        image_paths: List[str],
        matches: Dict
    ) -> Dict[int, np.ndarray]:
        """Collect unique keypoints for each image from matches."""
        image_keypoints = {}
        
        for (idx1, idx2), match_data in matches.items():
            # Add keypoints for image 1
            if idx1 not in image_keypoints:
                image_keypoints[idx1] = []
            image_keypoints[idx1].append(match_data['keypoints1'])
            
            # Add keypoints for image 2
            if idx2 not in image_keypoints:
                image_keypoints[idx2] = []
            image_keypoints[idx2].append(match_data['keypoints2'])
        
        # Concatenate and remove duplicates
        for idx in image_keypoints:
            if not image_keypoints[idx]:  # Skip if no keypoints
                continue
            all_kpts = np.vstack(image_keypoints[idx])
            logger.debug(f"all_kpts shape before unique: {all_kpts.shape}")
            # np.unique with axis=0 preserves shape
            unique_kpts = np.unique(all_kpts, axis=0)
            logger.debug(f"unique_kpts shape: {unique_kpts.shape}")
            # Ensure keypoints are 2D
            if len(unique_kpts.shape) == 1:
                unique_kpts = unique_kpts.reshape(-1, 2)
            image_keypoints[idx] = unique_kpts
        
        return image_keypoints
    
    def _create_camera(self, db, image_path: str) -> int:
        """Create camera entry in database."""
        from PIL import Image
        
        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Estimate focal length
        focal = self._estimate_focal_length(image_path, width, height)
        
        # Camera parameters based on model
        if self.config['camera_model'] == 'simple-pinhole':
            model = 0
            params = np.array([focal, width / 2, height / 2])
        elif self.config['camera_model'] == 'pinhole':
            model = 1
            params = np.array([focal, focal, width / 2, height / 2])
        elif self.config['camera_model'] == 'simple-radial':
            model = 2
            params = np.array([focal, width / 2, height / 2, 0.1])
        else:
            raise ValueError(f"Unknown camera model: {self.config['camera_model']}")
        
        return db.add_camera(model, width, height, params)
    
    def _estimate_focal_length(
        self,
        image_path: str,
        width: int,
        height: int
    ) -> float:
        """Estimate focal length from EXIF or use prior."""
        from PIL import Image, ExifTags
        
        # Try to get from EXIF
        try:
            img = Image.open(image_path)
            exif = img.getexif()
            
            if exif:
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, None)
                    if tag_name == 'FocalLengthIn35mmFilm':
                        focal_35mm = float(value)
                        max_size = max(width, height)
                        return focal_35mm / 35.0 * max_size
        except:
            pass
        
        # Use prior if EXIF not available
        FOCAL_PRIOR = 1.2
        return FOCAL_PRIOR * max(width, height)
    
    def _import_matches(
        self,
        db,
        matches: Dict,
        image_id_map: Dict[int, int]
    ):
        """Import matches to database."""
        # First, we need to rebuild the keypoint lists to create index mappings
        image_keypoints = self._collect_image_keypoints(
            [None] * len(image_id_map), matches
        )
        
        # Now process each match
        for (idx1, idx2), match_data in matches.items():
            image_id1 = image_id_map[idx1]
            image_id2 = image_id_map[idx2]
            
            # Get the matched keypoints
            kpts1 = match_data['keypoints1']
            kpts2 = match_data['keypoints2']
            
            # Get the consolidated keypoint lists for both images
            all_kpts1 = image_keypoints[idx1]
            all_kpts2 = image_keypoints[idx2]
            
            # Find indices of matched keypoints in the consolidated lists
            indices1 = []
            indices2 = []
            
            for i in range(len(kpts1)):
                # Find index of kpts1[i] in all_kpts1
                distances1 = np.sum((all_kpts1 - kpts1[i])**2, axis=1)
                idx1 = np.argmin(distances1)
                
                # Find index of kpts2[i] in all_kpts2
                distances2 = np.sum((all_kpts2 - kpts2[i])**2, axis=1)
                idx2 = np.argmin(distances2)
                
                indices1.append(idx1)
                indices2.append(idx2)
            
            # Create match indices array
            if indices1:  # Only add if there are matches
                match_indices = np.column_stack([indices1, indices2])
                db.add_matches(image_id1, image_id2, match_indices)
    
    def run_geometric_verification(self, database_path: Path):
        """Run geometric verification on matches."""
        import pycolmap
        
        logger.info("Running geometric verification...")
        
        # Use COLMAP's match_exhaustive to perform geometric verification
        # This automatically estimates two-view geometries and populates the database
        try:
            pycolmap.match_exhaustive(str(database_path))
            logger.info("Geometric verification completed successfully")
        except Exception as e:
            logger.error(f"Geometric verification failed: {e}")
            raise
    
    def get_mapper_options(self) -> Dict:
        """Get COLMAP mapper options from config."""
        import pycolmap
        
        options = pycolmap.IncrementalMapperOptions()
        options.min_model_size = self.config['min_model_size']
        options.ba_refine_focal = self.config['ba_refine_focal']
        options.ba_refine_principal = self.config['ba_refine_principal']
        options.ba_refine_distortion = self.config['ba_refine_distortion']
        options.ba_local_max_refinements = self.config['ba_local_max_refinements']
        options.ba_global_max_refinements = self.config['ba_global_max_refinements']
        
        return options
    
    def run_reconstruction(
        self,
        database_path: Path,
        image_path: Path,
        output_path: Path,
        min_num_matches: int = 15,
        exhaustive_matching: bool = True
    ) -> Dict[int, Any]:
        """
        Run COLMAP incremental reconstruction.
        
        Args:
            database_path: Path to COLMAP database
            image_path: Path to image directory
            output_path: Path for reconstruction output
            min_num_matches: Minimum number of matches for valid image pairs
            exhaustive_matching: If True, run exhaustive matching verification
            
        Returns:
            Reconstruction results
        """
        logger.info("Running COLMAP reconstruction")
        
        # Clear existing output directory if it exists
        import shutil
        if os.path.exists(output_path):
            logger.info(f"Clearing existing output directory: {output_path}")
            shutil.rmtree(output_path)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Configure pipeline options
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        # Configure mapper options through the pipeline options
        mapper_options = pipeline_options.mapper
        
        # Run exhaustive matching if requested
        if exhaustive_matching:
            logger.info("Running exhaustive matching verification")
            pycolmap.match_exhaustive(str(database_path))
        
        # Run incremental mapping
        # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        maps = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(output_path),
            options=pipeline_options
        )
        
        logger.info(f"Reconstruction complete. Found {len(maps)} models")
        
        # Log reconstruction statistics
        for idx, reconstruction in maps.items():
            num_images = len(reconstruction.images)
            num_points = len(reconstruction.points3D)
            logger.info(f"Model {idx}: {num_images} images, {num_points} 3D points")
        
        return maps
    
    def export_reconstruction(
        self,
        reconstruction: Any,
        output_path: str,
        format: str = 'ply'
    ) -> None:
        """
        Export reconstruction to various formats.
        
        Args:
            reconstruction: COLMAP reconstruction object
            output_path: Output file/directory path
            format: Export format ('ply', 'nvm', 'bundler', 'vrml')
        """
        try:
            if format == 'ply':
                # For PLY export, create directory and export
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                # COLMAP's write_text expects a directory, not a file path
                reconstruction.write_text(output_dir)
                logger.info(f"Exported reconstruction to {output_dir} in text format")
            else:
                logger.warning(f"Export format {format} not implemented, skipping export")
        except Exception as e:
            logger.error(f"Failed to export reconstruction: {e}")
    
    def import_features_from_h5(
        self,
        image_dir: str,
        feature_dir: str,
        database_path: str,
        camera_model: str = None,
        single_camera: bool = None
    ) -> Dict[str, int]:
        """
        Import features from h5 files to COLMAP database.
        
        This is an alternative method that imports from h5 files
        similar to the original image_matcher.py implementation.
        
        Args:
            image_dir: Directory containing images
            feature_dir: Directory containing h5 feature files
            database_path: Path to COLMAP database
            camera_model: Camera model to use
            single_camera: Whether to use single camera for all images
            
        Returns:
            Mapping from filename to image ID
        """
        camera_model = camera_model or self.config['camera_model']
        single_camera = single_camera if single_camera is not None else self.config['single_camera']
        
        # Create database
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        
        # Import keypoints
        keypoint_file = h5py.File(os.path.join(feature_dir, 'keypoints.h5'), 'r')
        
        camera_id = None
        fname_to_id = {}
        
        logger.info("Importing keypoints to COLMAP database")
        
        from tqdm import tqdm
        for filename in tqdm(list(keypoint_file.keys()), desc="Importing features"):
            keypoints = np.array(keypoint_file[filename])
            
            # Check if image exists
            image_path = os.path.join(image_dir, filename)
            if not os.path.isfile(image_path):
                logger.error(f"Image not found: {image_path}")
                continue
            
            # Create camera if needed
            if camera_id is None or not single_camera:
                camera_id = self._create_camera_db(db, image_path, camera_model)
            
            # Add image
            image_id = db.add_image(filename, camera_id)
            fname_to_id[filename] = image_id
            
            # Add keypoints
            db.add_keypoints(image_id, keypoints)
        
        keypoint_file.close()
        
        # Import matches
        match_file = h5py.File(os.path.join(feature_dir, 'matches.h5'), 'r')
        
        added = set()
        n_keys = len(match_file.keys())
        n_total = (n_keys * (n_keys - 1)) // 2
        
        logger.info("Importing matches to COLMAP database")
        
        with tqdm(total=n_total, desc="Importing matches") as pbar:
            for key1 in match_file.keys():
                group = match_file[key1]
                for key2 in group.keys():
                    if key1 not in fname_to_id or key2 not in fname_to_id:
                        logger.warning(f"Skipping match {key1} - {key2}: image not in database")
                        continue
                    
                    id1 = fname_to_id[key1]
                    id2 = fname_to_id[key2]
                    
                    pair_id = image_ids_to_pair_id(id1, id2)
                    if pair_id in added:
                        logger.warning(f"Duplicate pair: {key1} - {key2}")
                        continue
                    
                    matches = np.array(group[key2])
                    db.add_matches(id1, id2, matches)
                    
                    added.add(pair_id)
                    pbar.update(1)
        
        match_file.close()
        
        db.commit()
        db.close()
        
        return fname_to_id
    
    def _create_camera_db(self, db: COLMAPDatabase, image_path: str, camera_model: str) -> int:
        """
        Create camera entry in database (variant for direct db access).
        
        Args:
            db: COLMAP database connection
            image_path: Path to image
            camera_model: Camera model type
            
        Returns:
            Camera ID
        """
        from PIL import Image
        
        image = Image.open(image_path)
        width, height = image.size
        
        focal = get_focal_length(image_path)
        
        if camera_model == 'simple-pinhole':
            model = 0  # simple pinhole
            param_arr = np.array([focal, width / 2, height / 2])
        elif camera_model == 'pinhole':
            model = 1  # pinhole
            param_arr = np.array([focal, focal, width / 2, height / 2])
        elif camera_model == 'simple-radial':
            model = 2  # simple radial
            param_arr = np.array([focal, width / 2, height / 2, 0.1])
        elif camera_model == 'opencv':
            model = 4  # opencv
            param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
        else:
            raise ValueError(f"Unknown camera model: {camera_model}")
             
        return db.add_camera(model, width, height, param_arr)


class COLMAPDatabase(sqlite3.Connection):
    """
    COLMAP database interface.
    
    Based on COLMAP's database.py script.
    """
    
    @staticmethod
    def connect(database_path: str):
        """Connect to COLMAP database."""
        return sqlite3.connect(database_path, factory=COLMAPDatabase)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_tables()
    
    def create_tables(self):
        """Create necessary COLMAP tables."""
        # Create cameras table
        self.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB,
                prior_focal_length INTEGER NOT NULL
            )
        """)
        
        # Create images table
        self.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL,
                FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            )
        """)
        
        # Create keypoints table
        self.execute("""
            CREATE TABLE IF NOT EXISTS keypoints (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            )
        """)
        
        # Create matches table
        self.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB
            )
        """)
        
        # Create descriptors table
        self.execute("""
            CREATE TABLE IF NOT EXISTS descriptors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            )
        """)
        
        # Create two_view_geometries table
        self.execute("""
            CREATE TABLE IF NOT EXISTS two_view_geometries (
                pair_id INTEGER PRIMARY KEY NOT NULL,
                rows INTEGER NOT NULL,
                cols INTEGER NOT NULL,
                data BLOB,
                config INTEGER NOT NULL,
                F BLOB,
                E BLOB,
                H BLOB
            )
        """)
        
        self.commit()
    
    def add_camera(
        self,
        model: int,
        width: int,
        height: int,
        params: np.ndarray,
        prior_focal_length: bool = False,
        camera_id: Optional[int] = None
    ) -> int:
        """Add camera to database."""
        params = np.asarray(params, np.float64)
        
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params), prior_focal_length)
        )
        return cursor.lastrowid
    
    def add_image(
        self,
        name: str,
        camera_id: int,
        prior_q: np.ndarray = None,
        prior_t: np.ndarray = None,
        image_id: Optional[int] = None
    ) -> int:
        """Add image to database."""
        prior_q = prior_q if prior_q is not None else np.array([1.0, 0.0, 0.0, 0.0])
        prior_t = prior_t if prior_t is not None else np.zeros(3)
        
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, 
             prior_q[0], prior_q[1], prior_q[2], prior_q[3],
             prior_t[0], prior_t[1], prior_t[2])
        )
        return cursor.lastrowid
    
    def add_keypoints(self, image_id: int, keypoints: np.ndarray):
        """Add keypoints to database."""
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]
        
        keypoints = np.asarray(keypoints, np.float32)
        
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),)
        )
    
    def add_descriptors(self, image_id: int, descriptors: np.ndarray):
        """Add descriptors to database."""
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),)
        )
    
    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray):
        """Add matches to database."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),)
        )
    
    def add_two_view_geometry(self, image_id1: int, image_id2: int,
                            matches: np.ndarray,
                            F: np.ndarray = None,
                            E: np.ndarray = None,
                            H: np.ndarray = None,
                            config: int = 2) -> None:
        """Add two-view geometry for an image pair."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
        
        F = F if F is not None else np.eye(3)
        E = E if E is not None else np.eye(3)
        H = H if H is not None else np.eye(3)
        
        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H))
        )