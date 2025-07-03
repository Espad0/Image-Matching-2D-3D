"""
COLMAP interface for 3D reconstruction with proper geometric verification.

This module provides a clean interface to COLMAP functionality,
handling database creation, feature import, and reconstruction.
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import h5py

logger = logging.getLogger(__name__)


class ColmapInterface:
    """
    Interface for COLMAP database and reconstruction operations.
    
    This class handles:
    - Creating and managing COLMAP databases
    - Importing features and matches
    - Running geometric verification via match_exhaustive
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
            # Round to remove floating point duplicates
            all_kpts_rounded = np.round(all_kpts).astype(np.float32)
            # np.unique with axis=0 preserves shape
            unique_kpts = np.unique(all_kpts_rounded, axis=0)
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
        elif self.config['camera_model'] == 'opencv':
            model = 4
            params = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])
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
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        # Try to get from EXIF
        try:
            img = Image.open(image_path)
            exif = img.getexif()
            
            if exif:
                focal_35mm = None
                for tag, value in exif.items():
                    tag_name = TAGS.get(tag, None)
                    if tag_name == 'FocalLengthIn35mmFilm':
                        focal_35mm = float(value)
                        break
                
                if focal_35mm is not None:
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
            
            # Round keypoints for matching
            kpts1_rounded = np.round(kpts1).astype(np.float32)
            kpts2_rounded = np.round(kpts2).astype(np.float32)
            all_kpts1_rounded = np.round(all_kpts1).astype(np.float32)
            all_kpts2_rounded = np.round(all_kpts2).astype(np.float32)
            
            # Find indices of matched keypoints in the consolidated lists
            indices1 = []
            indices2 = []
            
            for i in range(len(kpts1)):
                # Find index of kpts1[i] in all_kpts1
                distances1 = np.sum((all_kpts1_rounded - kpts1_rounded[i])**2, axis=1)
                idx1 = np.argmin(distances1)
                
                # Find index of kpts2[i] in all_kpts2
                distances2 = np.sum((all_kpts2_rounded - kpts2_rounded[i])**2, axis=1)
                idx2 = np.argmin(distances2)
                
                indices1.append(idx1)
                indices2.append(idx2)
            
            # Create match indices array
            if indices1:  # Only add if there are matches
                match_indices = np.column_stack([indices1, indices2])
                db.add_matches(image_id1, image_id2, match_indices)
    
    def run_geometric_verification(self, database_path: Path):
        """Run geometric verification on matches using pycolmap.match_exhaustive."""
        import pycolmap
        
        logger.info("Running geometric verification...")
        
        # Use COLMAP's match_exhaustive which performs geometric verification
        # This is what the Kaggle notebook does and it works reliably
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
        
        # Create name index for faster lookups
        self.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)
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
        params_blob = params.astype(np.float64).tobytes()
        
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, params_blob, prior_focal_length)
        )
        return cursor.lastrowid
    
    def add_image(
        self,
        name: str,
        camera_id: int,
        prior_q: np.ndarray = np.zeros(4),
        prior_t: np.ndarray = np.zeros(3),
        image_id: Optional[int] = None
    ) -> int:
        """Add image to database."""
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2])
        )
        return cursor.lastrowid
    
    def add_keypoints(self, image_id: int, keypoints: np.ndarray):
        """Add keypoints to database."""
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]
        
        keypoints = np.asarray(keypoints, np.float32)
        keypoints_blob = keypoints.tobytes()
        
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_blob)
        )
    
    def add_descriptors(self, image_id: int, descriptors: np.ndarray):
        """Add descriptors to database."""
        assert len(descriptors.shape) == 2
        assert descriptors.dtype == np.uint8
        
        descriptors_blob = descriptors.tobytes()
        
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id, descriptors.shape[0], descriptors.shape[1], descriptors_blob)
        )
    
    def add_matches(self, image_id1: int, image_id2: int, matches: np.ndarray):
        """Add matches to database."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
            image_id1, image_id2 = image_id2, image_id1
        
        pair_id = self._image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        matches_blob = matches.tobytes()
        
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1], matches_blob)
        )
    
    def add_two_view_geometry(
        self,
        image_id1: int,
        image_id2: int,
        matches: np.ndarray,
        F: np.ndarray = np.eye(3),
        E: np.ndarray = np.eye(3),
        H: np.ndarray = np.eye(3),
        config: int = 2
    ):
        """Add two-view geometry to database."""
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2
        
        if image_id1 > image_id2:
            matches = matches[:, ::-1]
            image_id1, image_id2 = image_id2, image_id1
        
        pair_id = self._image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id, matches.shape[0], matches.shape[1], 
             matches.tobytes(), config,
             F.tobytes(), E.tobytes(), H.tobytes())
        )
    
    def _image_ids_to_pair_id(self, image_id1: int, image_id2: int) -> int:
        """Convert image IDs to pair ID."""
        MAX_IMAGE_ID = 2**31 - 1
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        return image_id1 * MAX_IMAGE_ID + image_id2