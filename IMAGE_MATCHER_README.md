# Image Matching and 3D Reconstruction Pipeline

A comprehensive pipeline for image matching and 3D reconstruction using state-of-the-art deep learning models (SuperGlue and LoFTR) with integrated COLMAP reconstruction. This tool provides an end-to-end solution from image matching to 3D model generation.

## Features

- **Multiple Matching Methods**: Choose between LoFTR+SuperGlue (recommended) or SuperGlue-only
- **Test-Time Augmentation (TTA)**: Improves matching robustness with various augmentations
- **Smart Image Pairing**: Efficiently selects image pairs using global descriptors
- **Batch Processing**: Handles large datasets with progress tracking
- **Integrated COLMAP Pipeline**: Complete 3D reconstruction with a single command
- **Multiple Export Formats**: Export reconstructions as PLY, NVM, Bundler, or VRML
- **Automatic Camera Calibration**: Extracts focal length from EXIF or uses intelligent priors

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Vision3D.git
cd Vision3D
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install SuperGlue** (required separately):
```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
# Add to Python path or copy the 'superglue' folder to your project
```

4. **Download model weights**:
- LoFTR weights: Place `loftr_outdoor.ckpt` in `kornia/` directory
- EfficientNet weights (optional): Place in `efficientnet/` directory

## Usage

### Basic Usage

Match images in a directory:
```bash
python image_matcher.py --input_dir ./images --output_dir ./features
```

**Run complete pipeline with 3D reconstruction:**
```bash
python image_matcher.py --input_dir ./images --output_dir ./features --run_reconstruction
```

### Advanced Options

**Use SuperGlue only (faster):**
```bash
python image_matcher.py --input_dir ./images --output_dir ./features --method superglue
```

**Full pipeline with custom camera model:**
```bash
python image_matcher.py --input_dir ./images --output_dir ./features --run_reconstruction --camera_model pinhole
```

**Export reconstruction in different format:**
```bash
python image_matcher.py --input_dir ./images --output_dir ./features --run_reconstruction --export_format nvm
```

**Use single camera for all images:**
```bash
python image_matcher.py --input_dir ./images --output_dir ./features --run_reconstruction --single_camera
```

**Adjust similarity threshold for pair selection:**
```bash
python image_matcher.py --input_dir ./images --similarity_threshold 0.7
```

**Use exhaustive matching for small datasets:**
```bash
python image_matcher.py --input_dir ./images --exhaustive
```

**Specify computation device:**
```bash
python image_matcher.py --input_dir ./images --device cuda  # or cpu
```

### Command-Line Arguments

#### Matching Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | Required | Directory containing input images |
| `--output_dir` | `./features` | Directory for output features and matches |
| `--method` | `loftr_superglue` | Matching method (`loftr_superglue` or `superglue`) |
| `--device` | `auto` | Device for computation (`auto`, `cuda`, or `cpu`) |
| `--similarity_threshold` | `0.6` | Similarity threshold for image pair selection |
| `--min_pairs` | `20` | Minimum number of pairs per image |
| `--min_matches` | `15` | Minimum matches required to save a pair |
| `--exhaustive` | `False` | Use exhaustive matching (all pairs) |
| `--log_level` | `INFO` | Logging level |

#### Reconstruction Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--run_reconstruction` | `False` | Run COLMAP reconstruction after matching |
| `--camera_model` | `simple-radial` | Camera model (`simple-pinhole`, `pinhole`, `simple-radial`, `radial`, `opencv`) |
| `--single_camera` | `False` | Use single camera for all images |
| `--skip_geometric_verification` | `False` | Skip geometric verification in COLMAP |
| `--export_format` | `ply` | Export format (`ply`, `nvm`, `bundler`, `vrml`) |

## Output Format

The pipeline generates the following files in the output directory:

### Matching Output
- `keypoints.h5`: Unique keypoints for each image
- `matches.h5`: Pairwise matches between images
- `matches_loftr.h5`: Raw matches before processing (temporary)

### Reconstruction Output (when `--run_reconstruction` is used)
- `colmap.db`: COLMAP database with features and matches
- `colmap_reconstruction/`: Directory containing reconstruction results
  - `0/`: First reconstruction model (binary COLMAP format)
  - `model_0.ply`: 3D point cloud (or other format as specified)
  - `model_0_stats.txt`: Reconstruction statistics

## Pipeline Overview

1. **Image Loading**: Reads images from the input directory
2. **Pair Generation**: 
   - Computes global descriptors using EfficientNet
   - Selects similar image pairs based on descriptor distance
   - Falls back to exhaustive matching for small datasets
3. **Feature Matching**:
   - Extracts and matches features using selected method
   - Applies Test-Time Augmentation for robustness
   - Combines matches from multiple scales
4. **Post-processing**:
   - Removes duplicate keypoints
   - Ensures one-to-one matching constraints
   - Saves in COLMAP-compatible format
5. **3D Reconstruction** (optional):
   - Creates COLMAP database with features and matches
   - Extracts camera intrinsics from EXIF or estimates them
   - Runs incremental Structure-from-Motion
   - Exports 3D model in various formats

## Performance Tips

- For **small datasets** (<20 images): Use exhaustive matching
- For **medium datasets** (20-400 images): Use default settings
- For **large datasets** (>400 images): Consider SuperGlue-only method
- Adjust `--similarity_threshold` based on your scene:
  - Lower values (0.4-0.5): More selective, fewer pairs
  - Higher values (0.7-0.8): More pairs, better coverage

## Examples

### Quick 3D Reconstruction
```bash
# For a small dataset (will use exhaustive matching automatically)
python image_matcher.py --input_dir ./vacation_photos --run_reconstruction

# For a large dataset with specific settings
python image_matcher.py \
    --input_dir ./drone_images \
    --output_dir ./reconstruction \
    --method superglue \
    --similarity_threshold 0.5 \
    --run_reconstruction \
    --camera_model opencv \
    --export_format ply
```

### Programmatic Usage

```python
# Run matching only
from image_matcher import match_images_loftr_superglue, get_image_pairs_shortlist

# Get image pairs
pairs = get_image_pairs_shortlist(image_files, similarity_threshold=0.6)

# Run matching
match_images_loftr_superglue(image_files, pairs, feature_dir='./output')

# Run reconstruction
from image_matcher import COLMAPDatabase, import_features_to_colmap, run_colmap_reconstruction

db = COLMAPDatabase.connect('output/colmap.db')
db.create_tables()
fname_to_id = import_features_to_colmap(db, './images', './output')
maps = run_colmap_reconstruction('output/colmap.db', './images', './output/reconstruction')
```

## Troubleshooting

**Out of Memory Errors**:
- Reduce image size with smaller `--similarity_threshold`
- Use `--method superglue` instead of `loftr_superglue`
- Process on CPU with `--device cpu`

**Too Few Matches**:
- Lower `--min_matches` threshold
- Increase `--similarity_threshold` for more pairs
- Check image quality and overlap

**Model Weights Not Found**:
- Ensure model files are in correct directories
- Download from respective repositories
- Check file permissions

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sarlin2020superglue,
  title={SuperGlue: Learning Feature Matching with Graph Neural Networks},
  author={Sarlin, Paul-Edouard and DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{sun2021loftr,
  title={LoFTR: Detector-Free Local Feature Matching with Transformers},
  author={Sun, Jiaming and Shen, Zehong and Wang, Yuang and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2021}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- SuperGlue: [MagicLeap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
- LoFTR: [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR)
- Kornia: [kornia/kornia](https://github.com/kornia/kornia)