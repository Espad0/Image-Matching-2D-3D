"""Test LoFTR with 1024 image size as used in the pipeline."""
import torch
import logging
import time
from pathlib import Path
from vision3d.models.loftr import LoFTRMatcher

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get test images
image_dir = Path("examples/images")
image_paths = list(image_dir.glob("*.png"))[:2]
img1_path = str(image_paths[0])
img2_path = str(image_paths[1])

print(f"Testing with images: {img1_path} and {img2_path}")

# Test with pipeline configuration
device = torch.device('cpu')
config = {
    'image_resize': 1024,  # Same as pipeline
    'confidence_threshold': 0.2,  # Same as pipeline
    'use_tta': True,  # Same as pipeline
    'tta_variants': ['orig', 'flip_lr'],  # Same as pipeline
    'match_threshold': 0.2,
    'max_keypoints': 8192
}

print("Creating LoFTR matcher with pipeline config...")
matcher = LoFTRMatcher(device=device, config=config, pretrained='outdoor')

print("Matching images with 1024 size...")
start_time = time.time()

try:
    mkpts1, mkpts2, mconf = matcher.match_pair(img1_path, img2_path)
    elapsed = time.time() - start_time
    
    print(f"\nSuccess! Matching completed in {elapsed:.2f} seconds")
    print(f"Found {len(mkpts1)} matches")
    print(f"Average confidence: {mconf.mean():.3f}")
    
except KeyboardInterrupt:
    print("\nInterrupted! Process was taking too long")
except Exception as e:
    print(f"\nError during matching: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()