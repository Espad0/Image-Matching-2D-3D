"""Complete debugging of the Vision3D pipeline with visualization."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from vision3d import Vision3DPipeline
from vision3d.models import LoFTRMatcher
from vision3d.utils.visualization import visualize_matches

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test images
img1_path = "examples/images/3DOM_FBK_IMG_1556.png"
img2_path = "examples/images/3DOM_FBK_IMG_1552.png"

# Create output directory
output_dir = Path("debug_full_output")
output_dir.mkdir(exist_ok=True)

print("="*50)
print("STEP 1: Testing Direct LoFTR Matching")
print("="*50)

# Test direct matching
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize matcher
matcher = LoFTRMatcher(device=device, config={
    'image_resize': 1024,
    'confidence_threshold': 0.2,
    'use_tta': False  # Disable TTA for simplicity
})

# Match the pair
mkpts1, mkpts2, mconf = matcher.match_pair(img1_path, img2_path)
print(f"Direct matching found {len(mkpts1)} matches")
if len(mkpts1) > 0:
    print(f"Confidence: min={mconf.min():.3f}, max={mconf.max():.3f}, mean={mconf.mean():.3f}")

# Load images for visualization
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Visualize matches
if len(mkpts1) > 0:
    vis_matches = visualize_matches(
        img1, img2, mkpts1, mkpts2, mconf,
        save_path=str(output_dir / "direct_matches.png")
    )
    print(f"Saved direct matches visualization")

# Save match data
match_data = {
    'num_matches': len(mkpts1),
    'keypoints1': mkpts1.tolist() if len(mkpts1) > 0 else [],
    'keypoints2': mkpts2.tolist() if len(mkpts2) > 0 else [],
    'confidence': mconf.tolist() if len(mconf) > 0 else [],
    'confidence_stats': {
        'min': float(mconf.min()) if len(mconf) > 0 else 0,
        'max': float(mconf.max()) if len(mconf) > 0 else 0,
        'mean': float(mconf.mean()) if len(mconf) > 0 else 0
    }
}

with open(output_dir / 'direct_match_data.json', 'w') as f:
    json.dump(match_data, f, indent=2)

print("\n" + "="*50)
print("STEP 2: Testing Full Pipeline")
print("="*50)

# Run the full pipeline
pipeline = Vision3DPipeline(matcher_type='loftr')

try:
    reconstruction = pipeline.reconstruct(
        [img1_path, img2_path],
        output_dir="debug_pipeline_output"
    )
    
    print(f"\nPipeline Results:")
    print(f"Registered images: {len(reconstruction['images'])}")
    print(f"Reconstructed points: {len(reconstruction['points3D'])}")
    print(f"Cameras: {len(reconstruction['cameras'])}")
    
    # Save reconstruction data
    with open(output_dir / 'reconstruction_result.json', 'w') as f:
        json.dump({
            'num_images': len(reconstruction['images']),
            'num_points3D': len(reconstruction['points3D']),
            'num_cameras': len(reconstruction['cameras']),
            'statistics': reconstruction.get('statistics', {})
        }, f, indent=2)
        
except Exception as e:
    print(f"Pipeline failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("STEP 3: Checking COLMAP Database")
print("="*50)

# Check the COLMAP database
import sqlite3
db_path = "debug_pipeline_output/colmap.db"
if Path(db_path).exists():
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    
    # Count entries
    cursor.execute("SELECT COUNT(*) FROM cameras")
    num_cameras = cursor.fetchone()[0]
    print(f"Cameras in DB: {num_cameras}")
    
    cursor.execute("SELECT COUNT(*) FROM images")
    num_images = cursor.fetchone()[0]
    print(f"Images in DB: {num_images}")
    
    cursor.execute("SELECT COUNT(*) FROM keypoints")
    num_keypoints = cursor.fetchone()[0]
    print(f"Images with keypoints: {num_keypoints}")
    
    cursor.execute("SELECT COUNT(*) FROM matches")
    num_matches = cursor.fetchone()[0]
    print(f"Image pairs with matches: {num_matches}")
    
    cursor.execute("SELECT COUNT(*) FROM descriptors")
    num_descriptors = cursor.fetchone()[0]
    print(f"Images with descriptors: {num_descriptors}")
    
    # Get detailed match info
    cursor.execute("SELECT pair_id, rows FROM matches")
    matches = cursor.fetchall()
    for pair_id, num in matches:
        print(f"  Pair {pair_id}: {num} matches")
    
    db.close()
else:
    print("COLMAP database not found!")

print("\n" + "="*50)
print("STEP 4: Alternative - Using OpenCV SIFT")
print("="*50)

# Try with SIFT for comparison
sift = cv2.SIFT_create(nfeatures=5000)

# Detect keypoints and descriptors
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"SIFT found {len(kp1)} keypoints in image 1")
print(f"SIFT found {len(kp2)} keypoints in image 2")

# Match with FLANN
if des1 is not None and des2 is not None:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"SIFT found {len(good_matches)} good matches after ratio test")
    
    # Draw matches
    if len(good_matches) > 0:
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, good_matches[:100], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(str(output_dir / 'sift_matches.png'), img_matches)
        print("Saved SIFT matches visualization")

print("\n" + "="*50)
print("Summary saved to:", output_dir)
print("="*50)