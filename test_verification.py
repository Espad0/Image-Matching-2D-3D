"""Test script to verify the geometric verification fix works."""
import sqlite3
import numpy as np
from vision3d import Vision3DPipeline
from pathlib import Path
import logging

# Enable info logging
logging.basicConfig(level=logging.INFO)

# Get test images
image_dir = Path("examples/images")
image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
image_paths = [str(p) for p in image_paths][:2]  # Use only 2 images for quick test

print(f"Testing with {len(image_paths)} images")

# Run pipeline
pipeline = Vision3DPipeline(matcher_type='loftr')  # Use LoFTR for speed
reconstruction = pipeline.reconstruct(
    image_paths,
    output_dir="output_test"
)

# Check database contents
db_path = "output_test/colmap.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check matches
cursor.execute("SELECT COUNT(*) FROM matches")
match_count = cursor.fetchone()[0]
print(f"\nMatches in database: {match_count}")

# Check two_view_geometries (this is what was missing before)
cursor.execute("SELECT COUNT(*) FROM two_view_geometries")
geometry_count = cursor.fetchone()[0]
print(f"Two-view geometries in database: {geometry_count}")

# Check if reconstruction succeeded
print(f"\nReconstruction results:")
print(f"Registered images: {len(reconstruction['images'])} / {len(image_paths)}")
print(f"3D points: {len(reconstruction['points3D'])}")
print(f"Cameras: {len(reconstruction['cameras'])}")

conn.close()

# Print success/failure
if geometry_count > 0 and len(reconstruction['images']) > 0:
    print("\n✅ SUCCESS: Geometric verification is working!")
else:
    print("\n❌ FAILED: Geometric verification still not working")