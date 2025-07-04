from vision3d import Vision3DPipeline
from pathlib import Path
import glob
import logging

# Enable info logging only (less verbose)
logging.basicConfig(level=logging.INFO)

# Get all image files from the directory
image_dir = Path("examples/images")
image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
image_paths = [str(p) for p in image_paths]

print(f"Found {len(image_paths)} images")

# Create custom config with relaxed settings
custom_config = {
    'matching': {
        'image_resize': 640,  # Smaller for faster processing
        'min_matches': 8,     # Lower threshold
        'confidence_threshold': 0.1,  # More permissive
        'use_tta': False     # Disable TTA for speed
    },
    'reconstruction': {
        'min_model_size': 2,  # Allow 2-image reconstructions
        'max_reproj_error': 8.0,  # More permissive
        'min_triangulation_angle': 0.5,  # Lower angle requirement
        'filter_max_reproj_error': 8.0,
        'filter_min_tri_angle': 0.5
    }
}

pipeline = Vision3DPipeline(config=custom_config)

reconstruction = pipeline.reconstruct(
    image_paths,           
    output_dir="output_relaxed"
)

print(f"\nReconstruction Results:")
print(f"Registered {reconstruction['num_registered_images']} / {reconstruction['total_images']} images")
print(f"Reconstructed {reconstruction['num_points3D']} 3D points")