"""Simple test to isolate the issue."""
from vision3d import Vision3DPipeline
from pathlib import Path
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get test images
image_dir = Path("examples/images")
image_paths = list(image_dir.glob("*.png"))[:2]  # Just 2 images
image_paths = [str(p) for p in image_paths]

print(f"Testing with {len(image_paths)} images: {[p.split('/')[-1] for p in image_paths]}")

# Create pipeline with specific matcher
pipeline = Vision3DPipeline(matcher_type='loftr')

# Run with timeout
try:
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Test timed out!")
    
    # Set 30 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    reconstruction = pipeline.reconstruct(
        image_paths,
        output_dir="output_test_simple"
    )
    
    signal.alarm(0)  # Disable alarm
    
    print("\nSuccess! Reconstruction completed.")
    print(f"Registered images: {len(reconstruction['images'])}")
    print(f"3D points: {len(reconstruction['points3D'])}")
    
except TimeoutError as e:
    print(f"\nError: {e}")
    print("The process is hanging, likely during geometric verification.")
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()