"""
Basic 3D reconstruction example.

This example demonstrates how to perform 3D reconstruction
from a collection of images using Vision3D.
"""

import argparse
from pathlib import Path
import logging
from vision3d import Vision3DPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run basic 3D reconstruction example."""
    parser = argparse.ArgumentParser(description='Basic 3D reconstruction example')
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--matcher',
        type=str,
        default='hybrid',
        choices=['loftr', 'superglue', 'hybrid'],
        help='Type of matcher to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation'
    )
    args = parser.parse_args()
    
    # Get image paths
    image_dir = Path(args.images)
    if not image_dir.exists():
        raise ValueError(f"Image directory {image_dir} does not exist")
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f'*{ext}')))
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Sort images by name for consistency
    image_paths = sorted([str(p) for p in image_paths])
    
    # Initialize pipeline
    logger.info(f"Initializing Vision3D pipeline with {args.matcher} matcher")
    pipeline = Vision3DPipeline(
        matcher_type=args.matcher,
        device=args.device
    )
    
    # Run reconstruction
    logger.info("Starting 3D reconstruction...")
    try:
        reconstruction = pipeline.reconstruct(
            image_paths,
            output_dir=args.output,
            verbose=True
        )
        
        # Print statistics
        stats = reconstruction['statistics']
        logger.info("\nReconstruction Statistics:")
        logger.info(f"  - Registered images: {stats['num_images']} / {len(image_paths)}")
        logger.info(f"  - 3D points: {stats['num_points3D']}")
        logger.info(f"  - Mean reprojection error: {stats['mean_reprojection_error']:.3f} px")
        logger.info(f"  - Mean track length: {stats['mean_track_length']:.1f}")
        
        # Export results
        logger.info("\nExporting results...")
        pipeline.export_results(
            reconstruction,
            args.output,
            formats=['ply', 'json']
        )
        
        logger.info(f"\nResults saved to {args.output}")
        logger.info("  - point_cloud.ply: 3D point cloud")
        logger.info("  - reconstruction.json: Camera poses and statistics")
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        raise


if __name__ == '__main__':
    main()