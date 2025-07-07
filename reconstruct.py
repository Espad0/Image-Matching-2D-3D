#!/usr/bin/env python3
"""
Vision3D command-line interface for 3D reconstruction.

This script provides backward compatibility with the original image_matcher.py
while using the new modular architecture.
"""

import argparse
import logging
import sys
from pathlib import Path

from vision3d import Vision3DPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the Vision3D CLI."""
    parser = argparse.ArgumentParser(
        description="Vision3D - 3D reconstruction from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction (auto-selects best method)
  python reconstruct.py --input_dir ./images --output_dir ./output
  
  # Use LoFTR + SuperGlue for best quality
  python reconstruct.py --input_dir ./images --output_dir ./output --method loftr_superglue
  
  # Use SuperGlue only for speed
  python reconstruct.py --input_dir ./images --output_dir ./output --method superglue
  
  # Adjust similarity threshold for pair selection
  python reconstruct.py --input_dir ./images --similarity_threshold 0.7
        """
    )
    
    # Backward compatibility with image_matcher.py arguments
    parser.add_argument('--input_dir', type=str, default='examples/images',
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./featureout',
                       help='Directory for output features and reconstruction')
    parser.add_argument('--method', type=str, default='auto',
                       choices=['loftr_superglue', 'superglue', 'loftr', 'auto'],
                       help='Matching method to use')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for computation')
    parser.add_argument('--similarity_threshold', type=float, default=0.5,
                       help='Similarity threshold for image pair selection')
    parser.add_argument('--min_pairs', type=int, default=35,
                       help='Minimum number of pairs per image')
    parser.add_argument('--min_matches', type=int, default=15,
                       help='Minimum number of matches to save a pair')
    parser.add_argument('--exhaustive', action='store_true',
                       help='Use exhaustive matching (all pairs)')
    parser.add_argument('--exhaustive_threshold', type=int, default=20,
                       help='Use exhaustive matching if fewer images than this')
    parser.add_argument('--large_dataset_threshold', type=int, default=400,
                       help='Threshold for switching to faster SuperGlue-only matching')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # COLMAP reconstruction options
    parser.add_argument('--skip_reconstruction', action='store_true',
                       help='Skip COLMAP reconstruction after matching')
    parser.add_argument('--camera_model', type=str, default='simple-radial',
                       choices=['simple-pinhole', 'pinhole', 'simple-radial', 'radial', 'opencv'],
                       help='Camera model for COLMAP')
    parser.add_argument('--single_camera', action='store_true',
                       help='Use single camera for all images')
    parser.add_argument('--export_format', type=str, default='ply',
                       choices=['ply', 'nvm', 'bundler', 'vrml'],
                       help='Export format for reconstruction')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate input directory
    if not Path(args.input_dir).is_dir():
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Map method names for backward compatibility
    if args.method == 'loftr_superglue':
        matcher_type = 'hybrid'
    elif args.method == 'auto':
        matcher_type = 'hybrid'
    else:
        matcher_type = args.method
    
    # Create configuration
    config = {
        'image_selection': {
            'min_pairs_per_image': args.min_pairs,
            'similarity_threshold': args.similarity_threshold,
            'exhaustive_if_less_than': args.exhaustive_threshold,
        },
        'feature_extraction': {
            'min_matches': args.min_matches,
            'large_dataset_threshold': args.large_dataset_threshold,
        },
        'reconstruction': {
            'camera_model': args.camera_model,
            'single_camera': args.single_camera,
            'min_num_matches': args.min_matches,
            'export_format': args.export_format,
        }
    }
    
    # Set device
    device = None if args.device == 'auto' else args.device
    
    # Initialize pipeline
    logger.info(f"Initializing Vision3D pipeline with {matcher_type} matcher")
    pipeline = Vision3DPipeline(
        config=config,
        matcher_type=matcher_type,
        device=device
    )
    
    # Run reconstruction
    logger.info(f"Starting 3D reconstruction pipeline...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Use exhaustive pairs if requested
        image_pairs = None
        if args.exhaustive:
            logger.info("Using exhaustive image pairing")
            from vision3d.utils.image_pairs import get_image_pairs_exhaustive
            image_paths = pipeline._get_image_paths(args.input_dir)
            image_pairs = get_image_pairs_exhaustive(image_paths)
        
        # Run pipeline
        results = pipeline.reconstruct(
            image_path=args.input_dir,
            output_dir=args.output_dir,
            image_pairs=image_pairs,
            skip_reconstruction=args.skip_reconstruction,
            verbose=True
        )
        
        logger.info("\nPipeline complete!")
        logger.info(f"Total time: {results['total_time']:.2f} seconds")
        
        # Print statistics
        if 'reconstructions' in results and results['reconstructions']:
            for idx, reconstruction in results['reconstructions'].items():
                logger.info(f"\nModel {idx}:")
                logger.info(f"  - Registered images: {len(reconstruction.images)}")
                logger.info(f"  - 3D points: {len(reconstruction.points3D)}")
        else:
            logger.info("\nFeatures saved to: " + args.output_dir)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()