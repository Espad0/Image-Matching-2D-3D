"""
Debug and visualize the Vision3D pipeline matching results.
This script saves intermediate results and visualizations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from vision3d import Vision3DPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory
output_dir = Path("debug_visualizations")
output_dir.mkdir(exist_ok=True)

# Test with two example images
image_paths = [
    "examples/images/3DOM_FBK_IMG_1556.png",
    "examples/images/3DOM_FBK_IMG_1552.png"
]

print(f"Testing with {len(image_paths)} images")

# Initialize pipeline
pipeline = Vision3DPipeline(matcher_type='loftr', config={
    'matching': {
        'image_resize': 1024,
        'min_matches': 15,
        'confidence_threshold': 0.2,
        'use_tta': False,  # Disable TTA for faster testing
    }
})

# Run reconstruction
try:
    reconstruction = pipeline.reconstruct(
        image_paths,
        output_dir="vision3d_debug_output",
        verbose=True
    )
    
    print(f"\n{'='*50}")
    print("RECONSTRUCTION RESULTS:")
    print(f"{'='*50}")
    print(f"Registered images: {len(reconstruction['images'])}/{len(image_paths)}")
    print(f"Reconstructed 3D points: {len(reconstruction['points3D'])}")
    print(f"Cameras: {len(reconstruction['cameras'])}")
    
    if reconstruction['statistics']:
        stats = reconstruction['statistics']
        print(f"\nStatistics:")
        print(f"  Mean reprojection error: {stats.get('mean_reprojection_error', 'N/A')}")
        print(f"  Mean track length: {stats.get('mean_track_length', 'N/A')}")
        
    # Save results
    with open(output_dir / 'reconstruction_summary.json', 'w') as f:
        json.dump({
            'num_input_images': len(image_paths),
            'num_registered_images': len(reconstruction['images']),
            'num_points3D': len(reconstruction['points3D']),
            'num_cameras': len(reconstruction['cameras']),
            'statistics': reconstruction.get('statistics', {})
        }, f, indent=2)
    
    # If successful, visualize the results
    if len(reconstruction['points3D']) > 0:
        from vision3d.utils.visualization import visualize_reconstruction
        
        fig = visualize_reconstruction(
            reconstruction,
            save_path=str(output_dir / 'reconstruction_3d.html'),
            show_cameras=True,
            show_points=True
        )
        print(f"\nSaved 3D visualization to {output_dir / 'reconstruction_3d.html'}")
        
except Exception as e:
    logger.error(f"Reconstruction failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*50}")
print(f"Debug output saved to: {output_dir}")
print(f"Pipeline output saved to: vision3d_debug_output")
print(f"{'='*50}")

# Also create a simple visualization of the matches
print("\nCreating match visualization...")

# Load the matches from the pipeline output
import sqlite3
db_path = Path("vision3d_debug_output/colmap.db")
if db_path.exists():
    db = sqlite3.connect(str(db_path))
    cursor = db.cursor()
    
    # Get match count
    cursor.execute("SELECT COUNT(*) FROM matches")
    num_match_pairs = cursor.fetchone()[0]
    
    cursor.execute("SELECT pair_id, rows FROM matches")
    matches = cursor.fetchall()
    
    print(f"\nDatabase contains:")
    print(f"  {num_match_pairs} image pair(s) with matches")
    for pair_id, num_matches in matches:
        print(f"  Pair {pair_id}: {num_matches} matches")
    
    db.close()
else:
    print("COLMAP database not found!")

print("\nDebug visualization complete!")