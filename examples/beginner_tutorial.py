"""
Beginner-Friendly 3D Reconstruction Tutorial

This script demonstrates how to use Vision3D to reconstruct a 3D model
from a collection of photos. Perfect for beginners!

Prerequisites:
- Python 3.8+
- GPU recommended (but CPU works too)
- Some photos of an object/scene from different angles

Author: Vision3D Team
"""

import os
import sys
sys.path.append('..')  # Add parent directory to path

from vision3d import Vision3DPipeline
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Main function demonstrating the complete 3D reconstruction pipeline.
    """
    
    print("🚀 Welcome to Vision3D - 3D Reconstruction Made Easy!")
    print("=" * 60)
    
    # ============================================================
    # STEP 1: Prepare Your Images
    # ============================================================
    print("\n📸 Step 1: Preparing images...")
    
    # Option A: Use example images (uncomment if you have them)
    # image_dir = "../data/example_images/"
    # images = [
    #     os.path.join(image_dir, f)
    #     for f in os.listdir(image_dir)
    #     if f.endswith(('.jpg', '.png', '.jpeg'))
    # ]
    
    # Option B: Use your own images (recommended)
    images = [
        'path/to/your/image1.jpg',
        'path/to/your/image2.jpg',
        'path/to/your/image3.jpg',
        # Add more images here...
    ]
    
    print(f"Found {len(images)} images")
    
    # Important tips for good reconstruction:
    # 1. Take photos from different angles (but with overlap)
    # 2. Ensure good lighting (avoid shadows)
    # 3. Keep the camera settings consistent
    # 4. More images = better reconstruction (aim for 20-100)
    
    # ============================================================
    # STEP 2: Create the Pipeline
    # ============================================================
    print("\n🔧 Step 2: Creating reconstruction pipeline...")
    
    # Create pipeline with default settings (recommended for beginners)
    pipeline = Vision3DPipeline(
        matcher_type='hybrid',  # Automatically chooses best method
        device=None  # Auto-selects GPU if available
    )
    
    # Advanced: Customize settings for your use case
    # custom_config = {
    #     'image_resize': 800,  # Smaller = faster (but less accurate)
    #     'matching': {
    #         'confidence_threshold': 0.3,  # Higher = fewer but better matches
    #         'min_matches': 20  # Minimum matches needed between image pairs
    #     }
    # }
    # pipeline = Vision3DPipeline(config=custom_config)
    
    print("Pipeline ready!")
    
    # ============================================================
    # STEP 3: Run 3D Reconstruction
    # ============================================================
    print("\n🏗️ Step 3: Running 3D reconstruction...")
    print("This may take a few minutes depending on the number of images.")
    print("The pipeline will:")
    print("  1. Find which images overlap")
    print("  2. Match features between overlapping images")
    print("  3. Reconstruct 3D structure")
    print("  4. Optimize the result")
    
    try:
        # Run reconstruction
        result = pipeline.reconstruct(
            images,
            output_dir='./my_reconstruction',  # Where to save results
            verbose=True  # Show progress
        )
        
        print("\n✅ Reconstruction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during reconstruction: {e}")
        print("Common issues:")
        print("  - Not enough images (need at least 3)")
        print("  - Images don't overlap enough")
        print("  - Images are too blurry or dark")
        return
    
    # ============================================================
    # STEP 4: Analyze Results
    # ============================================================
    print("\n📊 Step 4: Analyzing results...")
    
    # Print statistics
    stats = result['statistics']
    print(f"\nReconstruction Statistics:")
    print(f"  - Total images: {len(images)}")
    print(f"  - Reconstructed images: {stats['num_images']}")
    print(f"  - 3D points: {stats['num_points3D']:,}")
    print(f"  - Cameras: {stats['num_cameras']}")
    print(f"  - Avg reprojection error: {stats['mean_reprojection_error']:.2f} pixels")
    
    # Success rate
    success_rate = stats['num_images'] / len(images) * 100
    print(f"\n  Success rate: {success_rate:.1f}%")
    
    if success_rate < 50:
        print("  ⚠️  Low success rate. Try:")
        print("     - Taking more photos with better overlap")
        print("     - Ensuring consistent lighting")
        print("     - Using higher resolution images")
    elif success_rate < 80:
        print("  👍 Good reconstruction! Could be improved with more images.")
    else:
        print("  🎉 Excellent reconstruction!")
    
    # ============================================================
    # STEP 5: Export Results
    # ============================================================
    print("\n💾 Step 5: Exporting results...")
    
    # Export in multiple formats
    pipeline.export_results(
        result,
        output_path='./my_reconstruction/exports',
        formats=['ply', 'json']  # PLY for 3D viewers, JSON for data analysis
    )
    
    print("\nExported files:")
    print("  - point_cloud.ply: 3D point cloud (open in MeshLab, CloudCompare, etc.)")
    print("  - reconstruction.json: Camera poses and metadata")
    
    # ============================================================
    # STEP 6: Visualize Results (Optional)
    # ============================================================
    print("\n🎨 Step 6: Visualizing results...")
    
    # Simple visualization of camera positions
    visualize_cameras(result)
    
    print("\n🎉 All done! Your 3D reconstruction is ready.")
    print("Next steps:")
    print("  - Open point_cloud.ply in a 3D viewer")
    print("  - Try with more images for better results")
    print("  - Experiment with different settings")


def visualize_cameras(result):
    """
    Simple visualization of camera positions in 3D space.
    
    This helps you understand where each photo was taken from.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract camera positions
        positions = []
        for img_id, img_data in result['images'].items():
            # Camera position is -R^T * t
            R = img_data['R']
            t = img_data['t']
            pos = -R.T @ t
            positions.append(pos)
        
        if positions:
            positions = np.array(positions)
            
            # Plot camera positions
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c='red', marker='o', s=100, label='Cameras')
            
            # Plot origin
            ax.scatter([0], [0], [0], c='black', marker='x', s=200, label='Origin')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Camera Positions in 3D Space')
            ax.legend()
            
            plt.savefig('./my_reconstruction/camera_positions.png')
            print("  Saved camera visualization to camera_positions.png")
        else:
            print("  No camera positions to visualize")
            
    except ImportError:
        print("  Skipping visualization (matplotlib not installed)")


def advanced_example():
    """
    Advanced example showing how to use specific matchers and custom settings.
    """
    print("\n🔬 Advanced Example: Custom Configuration")
    
    # Example 1: Force LoFTR for textureless scenes
    pipeline_loftr = Vision3DPipeline(
        matcher_type='loftr',
        config={
            'image_resize': 1440,  # Higher resolution for accuracy
            'matching': {
                'confidence_threshold': 0.4,
                'use_tta': True,  # Test-time augmentation
                'tta_variants': ['orig', 'flip_lr', 'rot_90']
            }
        }
    )
    
    # Example 2: Force SuperGlue for speed
    pipeline_superglue = Vision3DPipeline(
        matcher_type='superglue',
        config={
            'image_resize': 800,  # Lower resolution for speed
            'matching': {
                'min_matches': 30,  # Require more matches
                'max_keypoints': 4096  # Detect more keypoints
            }
        }
    )
    
    # Example 3: Custom pair selection for large datasets
    pipeline_large = Vision3DPipeline(
        config={
            'pair_selection': {
                'min_pairs': 50,  # Each image matched with 50 others
                'similarity_threshold': 0.7,  # Stricter similarity
                'exhaustive_if_less': 30  # Use all pairs if < 30 images
            }
        }
    )
    
    print("Advanced configurations created!")


if __name__ == "__main__":
    # Run the main tutorial
    main()
    
    # Uncomment to see advanced examples
    # advanced_example()
    
    print("\n📚 Want to learn more?")
    print("  - Read THEORY.md for the mathematical foundations")
    print("  - Check the API documentation")
    print("  - Join our community forum")
    print("\nHappy reconstructing! 🚀")