"""
Basic 3D reconstruction example.

This example demonstrates how to perform 3D reconstruction
from a collection of images using Vision3D.

Usage:
    python basic_reconstruction.py --images examples/images --output ./output
"""

import argparse
from pathlib import Path
import logging
import sys
import os

# Add parent directory to path to import vision3d
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vision3d import Vision3DPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run basic 3D reconstruction example."""
    parser = argparse.ArgumentParser(
        description='Basic 3D reconstruction example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction with default settings
  python basic_reconstruction.py --images ./images --output ./output
  
  # Use only LoFTR for matching
  python basic_reconstruction.py --images ./images --output ./output --matcher loftr
  
  # Force CPU computation
  python basic_reconstruction.py --images ./images --output ./output --device cpu
        """
    )
    parser.add_argument(
        '--images',
        type=str,
        default='examples/images',
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
        help='Type of matcher to use (default: hybrid)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use for computation (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--skip-reconstruction',
        action='store_true',
        help='Skip 3D reconstruction (only extract features)'
    )
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = None  # Let pipeline decide
    else:
        device = args.device
    
    # Initialize pipeline
    logger.info(f"Initializing Vision3D pipeline with {args.matcher} matcher")
    pipeline = Vision3DPipeline(
        matcher_type=args.matcher,
        device=device
    )
    
    # Run reconstruction
    logger.info("Starting 3D reconstruction pipeline...")
    logger.info(f"Input images: {args.images}")
    logger.info(f"Output directory: {args.output}")
    
    try:
        # Run the pipeline
        results = pipeline.reconstruct(
            image_path=args.images,
            output_dir=args.output,
            skip_reconstruction=args.skip_reconstruction,
            verbose=True
        )
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("Pipeline Results:")
        logger.info("="*50)
        logger.info(f"Total time: {results['total_time']:.2f} seconds")
        logger.info(f"Number of images: {results['num_images']}")
        
        if 'num_pairs' in results:
            logger.info(f"Image pairs selected: {results['num_pairs']}")
        
        if 'reconstructions' in results and results['reconstructions']:
            logger.info("\nReconstruction Statistics:")
            for idx, reconstruction in results['reconstructions'].items():
                logger.info(f"\nModel {idx}:")
                logger.info(f"  - Registered images: {len(reconstruction.images)}")
                logger.info(f"  - 3D points: {len(reconstruction.points3D)}")
                
                # Calculate mean reprojection error
                if len(reconstruction.points3D) > 0:
                    errors = [point.error for point in reconstruction.points3D.values()]
                    mean_error = sum(errors) / len(errors)
                    logger.info(f"  - Mean reprojection error: {mean_error:.3f} px")
        
        logger.info(f"\nResults saved to: {results['output_dir']}")
        
        if 'feature_path' in results:
            logger.info(f"  - Features: {results['feature_path']}")
        
        if 'reconstruction_path' in results:
            logger.info(f"  - 3D models: {results['reconstruction_path']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()