"""Test COLMAP directly with pycolmap."""

import numpy as np
import pycolmap
from pathlib import Path
import cv2

# Create output directory
output_dir = Path("test_colmap_output")
output_dir.mkdir(exist_ok=True)
database_path = output_dir / "database.db"

# Remove existing database
if database_path.exists():
    database_path.unlink()

# Test images
img1_path = "examples/images/3DOM_FBK_IMG_1556.png"
img2_path = "examples/images/3DOM_FBK_IMG_1552.png"

print("Creating COLMAP database...")

# Extract features using SIFT
sift = cv2.SIFT_create(nfeatures=5000)

# Process image 1
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
kp1, desc1 = sift.detectAndCompute(img1, None)
keypoints1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1], dtype=np.float32)
print(f"Image 1: {len(keypoints1)} keypoints")

# Process image 2  
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
kp2, desc2 = sift.detectAndCompute(img2, None)
keypoints2 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp2], dtype=np.float32)
print(f"Image 2: {len(keypoints2)} keypoints")

# Match features
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)

# Apply Lowe's ratio test
good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

print(f"Found {len(good_matches)} good matches")

# Create match array
matches_array = np.array([[m.queryIdx, m.trainIdx] for m in good_matches], dtype=np.uint32)

# Import to COLMAP using pycolmap
print("\nImporting to COLMAP database...")
import sqlite3

# Connect to database
db = sqlite3.connect(str(database_path))
cursor = db.cursor()

# Create tables
cursor.executescript("""
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL
);

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
);

CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB
);
""")

# Add cameras
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
focal = max(w1, h1) * 1.2

# Camera 1
params1 = np.array([focal, w1/2, h1/2, 0.0], dtype=np.float64)
cursor.execute(
    "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
    (1, 2, w1, h1, params1.tobytes(), 0)
)

# Camera 2
params2 = np.array([focal, w2/2, h2/2, 0.0], dtype=np.float64)
cursor.execute(
    "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
    (2, 2, w2, h2, params2.tobytes(), 0)
)

# Add images
cursor.execute(
    "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (1, Path(img1_path).name, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)
cursor.execute(
    "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (2, Path(img2_path).name, 2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
)

# Add keypoints
cursor.execute(
    "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
    (1, keypoints1.shape[0], keypoints1.shape[1], keypoints1.tobytes())
)
cursor.execute(
    "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
    (2, keypoints2.shape[0], keypoints2.shape[1], keypoints2.tobytes())
)

# Add descriptors
cursor.execute(
    "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
    (1, desc1.shape[0], desc1.shape[1], desc1.tobytes())
)
cursor.execute(
    "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
    (2, desc2.shape[0], desc2.shape[1], desc2.tobytes())
)

# Add matches
# Image pair ID for images 1 and 2
pair_id = 1 * 2147483647 + 2
cursor.execute(
    "INSERT INTO matches VALUES (?, ?, ?, ?)",
    (pair_id, matches_array.shape[0], matches_array.shape[1], matches_array.tobytes())
)

db.commit()
db.close()

print(f"Database created with {len(matches_array)} matches")

# Now run COLMAP reconstruction
print("\nRunning COLMAP reconstruction...")

# Run reconstruction with defaults
image_dir = Path(img1_path).parent

# Try using the simple reconstruction function
try:
    reconstruction = pycolmap.incremental_reconstruction(
        str(database_path),
        str(image_dir),
        str(output_dir)
    )
    print(f"\nReconstruction complete!")
    if hasattr(reconstruction, '__len__'):
        print(f"Number of reconstructions: {len(reconstruction)}")
        maps = reconstruction
    else:
        maps = [reconstruction] if reconstruction else []
except Exception as e:
    print(f"Reconstruction failed: {e}")
    maps = []

if maps is None:
    maps = []
print(f"Number of reconstructions: {len(maps)}")

if maps:
    reconstruction = maps[0]
    print(f"Images: {len(reconstruction.images)}")
    print(f"Points3D: {len(reconstruction.points3D)}")
    print(f"Cameras: {len(reconstruction.cameras)}")
else:
    print("No valid reconstruction found")
    
# Let's also check what COLMAP sees in the database
print("\n" + "="*50)
print("Checking database contents from COLMAP's perspective...")

db = pycolmap.Database(database_path)
print(f"Number of cameras: {db.num_cameras}")
print(f"Number of images: {db.num_images}")
print(f"Number of keypoints: {db.num_keypoints}")
print(f"Number of descriptors: {db.num_descriptors}")
print(f"Number of matches: {db.num_matches}")
print(f"Number of two-view geometries: {db.num_two_view_geometries}")

# List all images
print("\nImages in database:")
for image_id in db.image_ids:
    image = db.read_image(image_id)
    print(f"  Image {image_id}: {image.name}")
    
db.close()