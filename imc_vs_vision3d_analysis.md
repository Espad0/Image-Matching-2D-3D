# IMC Notebook vs Vision3D Pipeline Analysis

## 1. Matching Strategy

### IMC Notebook
```python
# For scenes with >= 400 pairs:
- Uses SuperGlue with 8096 keypoints
- Multiple resolutions: 1024 and 1440 pixels
- No LoFTR for large scenes

# For scenes with < 400 pairs:
- LoFTR at 1024 pixels
- SuperGlue at 1024 and 1440 pixels  
- Combines all matches (concatenates them)
```

### Vision3D
```python
# Uses hybrid approach:
- LoFTR with default resolution (640 pixels typically)
- Single pass matching
- No multi-resolution strategy
```

## 2. Test-Time Augmentation (TTA)

### IMC Notebook
- SuperGlue uses TTA groups: `[('orig', 'orig'), ('flip_lr', 'flip_lr')]`
- LoFTR uses TTA: `['orig', 'flip_lr']` by default
- Rotation augmentations available but not used in main pipeline

### Vision3D
- No TTA implementation
- Single forward pass only

## 3. Image Resolution

### IMC Notebook
- Matching at higher resolutions: 1024, 1440 pixels
- Preserves more detail for accurate matching

### Vision3D
- Default LoFTR resize: 640 pixels (from config)
- Lower resolution = less accurate keypoint localization

## 4. Match Aggregation

### IMC Notebook
```python
# Combines matches from multiple sources:
mkpts0 = np.concatenate([
    mkpts0_loftr,           # LoFTR forward
    mkpts0_superglue_1024,  # SuperGlue at 1024
    mkpts0_superglue_1440,  # SuperGlue at 1440  
    mkpts0_loftr_lr         # LoFTR reverse (img2->img1)
], axis=0)
```

### Vision3D
- Single matcher output
- No match aggregation from multiple methods/scales

## 5. Reconstruction Parameters

### IMC Notebook
```python
mapper_options = pycolmap.IncrementalMapperOptions()
mapper_options.min_model_size = 3
mapper_options.ba_local_max_refinements = 2
# mapper_options.ba_global_max_refinements = 20  # commented out
```

### Vision3D
- Uses default COLMAP parameters
- No custom optimization for speed/quality trade-off

## 6. Image Pair Selection

### IMC Notebook
- Uses global descriptors (EfficientNet-B7) for shortlisting
- Similarity threshold: 0.5
- Minimum pairs per image: 35

### Vision3D
- Exhaustive matching for small scenes
- No intelligent pair selection for larger scenes

## 7. Match Deduplication

### IMC Notebook
- Sophisticated deduplication after aggregating matches
- Ensures unique keypoints per image
- Handles forward/reverse matches properly

### Vision3D
- Basic duplicate removal in keypoint collection
- No cross-method deduplication

## Key Improvements for Vision3D

1. **Multi-resolution matching**: Process images at 1024 and 1440 pixels
2. **Test-time augmentation**: Add horizontal flip at minimum
3. **Match aggregation**: Combine LoFTR + SuperGlue results
4. **Bidirectional matching**: Run LoFTR in both directions
5. **Custom mapper options**: Optimize bundle adjustment iterations
6. **Better pair selection**: Implement global descriptor shortlisting