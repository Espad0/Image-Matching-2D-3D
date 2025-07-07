#!/usr/bin/env python3
from vision3d import Vision3DPipeline
import sys

# Default paths
input_dir = sys.argv[1] if len(sys.argv) > 1 else "examples/images"
output_dir = sys.argv[2] if len(sys.argv) > 2 else "./featureout"

# Create pipeline
pipeline = Vision3DPipeline()

# Run reconstruction
reconstruction = pipeline.reconstruct(
    input_dir,
    output_dir=output_dir
)

print(f"\nReconstruction complete! Output saved to: {output_dir}")