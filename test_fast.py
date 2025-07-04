from vision3d import Vision3DPipeline
from pathlib import Path
import logging
import time

# Enable info logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get all image files from the directory
image_dir = Path("examples/images")
image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
image_paths = [str(p) for p in image_paths]

logger.info(f"Found {len(image_paths)} images")

# Use LoFTR with TTA for a faster test
logger.info("\n" + "="*60)
logger.info("Running with: LoFTR + TTA (Faster option)")
logger.info("Description: Dense matching with test-time augmentation at 1024px")
logger.info("="*60)

# Create pipeline with LoFTR + TTA
pipeline = Vision3DPipeline(
    matcher_type='loftr',
    config={
        'matching': {
            'use_tta': True,
            'tta_variants': ['orig', 'flip_lr'],
            'image_resize': 1024,
            'confidence_threshold': 0.3
        }
    }
)

# Time the reconstruction
start_time = time.time()

# Run reconstruction
reconstruction = pipeline.reconstruct(
    image_paths,
    output_dir="output_loftr_tta"
)

elapsed = time.time() - start_time

# Print results
logger.info(f"\n{'='*60}")
logger.info("RECONSTRUCTION RESULTS")
logger.info(f"{'='*60}")
logger.info(f"Time elapsed: {elapsed:.2f} seconds")
logger.info(f"Images registered: {reconstruction['num_images']} / {len(image_paths)}")
logger.info(f"3D points: {reconstruction['num_points3D']}")

if reconstruction['statistics']:
    stats = reconstruction['statistics']
    logger.info(f"Mean reprojection error: {stats.get('mean_reprojection_error', 'N/A'):.2f} pixels")
    logger.info(f"Mean track length: {stats.get('mean_track_length', 'N/A'):.1f}")

logger.info("\n" + "="*60)
logger.info("NOTE: This used LoFTR with TTA for speed.")
logger.info("For best results (but slower), use matcher_type='multi'")
logger.info("="*60)