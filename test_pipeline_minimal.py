"""Minimal pipeline test with custom config to isolate issue."""
from vision3d import Vision3DPipeline
from pathlib import Path
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get test images
image_dir = Path("examples/images")
image_paths = list(image_dir.glob("*.png"))[:2]
image_paths = [str(p) for p in image_paths]

print(f"Testing with {len(image_paths)} images")

# Create custom config to disable TTA and reduce image size
config = {
    'matching': {
        'image_resize': 640,  # Smaller size
        'min_matches': 15,
        'confidence_threshold': 0.3,
        'use_tta': False,  # Disable TTA
        'max_keypoints': 2048
    }
}

# Create pipeline with custom config
pipeline = Vision3DPipeline(matcher_type='loftr', config=config)

try:
    print("Starting reconstruction...")
    reconstruction = pipeline.reconstruct(
        image_paths,
        output_dir="output_minimal",
        verbose=False  # Disable progress bars
    )
    
    print("\nSuccess! Reconstruction completed.")
    print(f"Registered images: {len(reconstruction['images'])}")
    print(f"3D points: {len(reconstruction['points3D'])}")
    
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()