# Quick Start Guide

Welcome to Vision3D! This guide will get you up and running in 5 minutes.

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
- Install Vision3D
- Run your first 3D reconstruction
- Understand the results
- Know where to learn more

## 📚 Prerequisites

- Basic Python knowledge
- Some photos of an object/scene (10-50 images work best)
- 10 minutes of your time

## 🚀 Installation (2 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/vision3d.git
cd vision3d

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install COLMAP (required for 3D reconstruction)
# Ubuntu/Debian:
sudo apt-get install colmap
# macOS:
brew install colmap
# Windows: Download from https://colmap.github.io/
```

## 🏃 Your First Reconstruction (3 minutes)

### Option 1: Use the Simple Script (Recommended for Beginners)

```python
from vision3d import Vision3DPipeline

# Create pipeline
pipeline = Vision3DPipeline()

# Your images
images = ['photo1.jpg', 'photo2.jpg', 'photo3.jpg']  # Add your images

# Run reconstruction
result = pipeline.reconstruct(images)

# Save results
pipeline.export_results(result, 'my_first_reconstruction/')
```

### Option 2: Use the Tutorial Script

```bash
# Run the beginner tutorial
python examples/beginner_tutorial.py
```

## 📊 Understanding Your Results

After reconstruction, you'll find:

```
my_first_reconstruction/
├── sparse/          # COLMAP reconstruction files
├── exports/
│   ├── point_cloud.ply    # 3D point cloud (open in MeshLab)
│   └── reconstruction.json # Camera poses and metadata
└── features/        # Intermediate matching results
```

### What the Numbers Mean

```
Reconstruction Statistics:
  - Reconstructed images: 8/10  # 8 out of 10 images were successfully placed
  - 3D points: 12,543          # Number of 3D points in your model
  - Avg error: 1.2 pixels      # How accurate the reconstruction is
```

**Success Metrics:**
- ✅ **>80% images reconstructed** = Great!
- ⚠️ **50-80% reconstructed** = Good, but could be better
- ❌ **<50% reconstructed** = Need better images

## 🎓 Learning Path

### Beginner (Start Here)
1. Read [THEORY.md](THEORY.md) - Understand the concepts
2. Run [beginner_tutorial.py](examples/beginner_tutorial.py) - Hands-on practice
3. Experiment with your own photos

### Intermediate
1. Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Understand the code
2. Try different matcher types (LoFTR vs SuperGlue)
3. Tune parameters for your use case

### Advanced
1. Read the research papers (see [THEORY.md](THEORY.md#further-reading))
2. Implement custom matchers
3. Contribute to the project!

## 🆘 Troubleshooting

### "Not enough matches found"
- **Cause**: Images don't overlap enough
- **Fix**: Take photos with 60-80% overlap

### "CUDA out of memory"
- **Cause**: Images too large for GPU
- **Fix**: Reduce image size: `config = {'image_resize': 800}`

### "Reconstruction failed"
- **Cause**: Usually poor image quality or not enough images
- **Fix**: 
  - Add more images
  - Ensure good lighting
  - Avoid blurry photos

## 📸 Tips for Better Results

### Taking Photos
1. **Overlap**: Each photo should overlap 60-80% with neighbors
2. **Coverage**: Capture from multiple angles
3. **Consistency**: Keep camera settings constant
4. **Quality**: Avoid motion blur and ensure good focus

### Recommended Patterns
- **Object**: Walk around it, taking photos every 10-15 degrees
- **Building**: Take photos along the facade with overlap
- **Room**: Stand in corners and center, capture all walls

[TODO: image showing good photo capture patterns]

## 🚀 Next Steps

1. **Visualize Your Model**: Download [MeshLab](https://www.meshlab.net/) to view your .ply file
2. **Read the Docs**: Check out [THEORY.md](THEORY.md) to understand how it works
3. **Join the Community**: Ask questions and share your reconstructions
4. **Experiment**: Try different scenes and settings

## 💡 Pro Tips

```python
# For indoor scenes (less texture)
pipeline = Vision3DPipeline(matcher_type='loftr')

# For large outdoor scenes (speed matters)
pipeline = Vision3DPipeline(matcher_type='superglue')

# For maximum quality (slow but accurate)
pipeline = Vision3DPipeline(
    config={
        'image_resize': 1440,
        'matching': {'use_tta': True}
    }
)
```

## 🎉 Congratulations!

You've just performed 3D reconstruction using state-of-the-art computer vision! 

**What you've accomplished:**
- Installed a complex computer vision pipeline
- Ran deep learning models (LoFTR/SuperGlue)
- Created a 3D model from 2D photos
- Learned the basics of 3D reconstruction

**Share your results!** Tag us on social media with #Vision3D

---

Need help? Check the [FAQ](docs/FAQ.md) or open an issue on GitHub!