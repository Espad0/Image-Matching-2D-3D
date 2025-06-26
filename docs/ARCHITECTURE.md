# Vision3D Architecture Guide

## Overview

This document explains the architecture of Vision3D in detail, helping developers understand how components work together.

[TODO: image showing high-level architecture diagram]

## Core Components

### 1. Pipeline Layer (`vision3d.core`)

The pipeline layer orchestrates the entire reconstruction process.

```
vision3d.core/
├── pipeline.py          # Main orchestrator
├── reconstruction.py    # 3D reconstruction engine
└── feature_extraction.py # Feature extraction utilities
```

**Key Classes:**
- `Vision3DPipeline`: Main entry point, coordinates all components
- `ReconstructionEngine`: Wraps COLMAP for 3D reconstruction
- `FeatureExtractor`: Base class for feature extraction

### 2. Models Layer (`vision3d.models`)

Deep learning models for feature detection and matching.

```
vision3d.models/
├── base.py       # Abstract base class for matchers
├── loftr.py      # LoFTR implementation
└── superglue.py  # SuperGlue implementation
```

**Design Pattern:** Strategy Pattern
- `BaseMatcher`: Defines the interface all matchers must implement
- Concrete implementations can be swapped without changing client code

### 3. Utils Layer (`vision3d.utils`)

Helper utilities and algorithms.

```
vision3d.utils/
├── image_pairs.py       # Smart pair selection
├── colmap_interface.py  # COLMAP database management  
├── io.py               # File I/O utilities
└── visualization.py    # Result visualization
```

## Data Flow

### 1. Image Loading and Preprocessing

```python
Images → Resize → Normalize → Tensor
```

- Images are resized to a maximum dimension (preserving aspect ratio)
- Pixel values normalized to [0, 1]
- Converted to PyTorch tensors for GPU processing

### 2. Image Pair Selection

```python
Images → Global Descriptors → Similarity Matrix → Selected Pairs
```

**Algorithm:**
1. Extract global descriptor for each image using CNN
2. Compute pairwise similarities
3. Select pairs above threshold
4. Ensure minimum pairs per image

### 3. Feature Matching Pipeline

```python
Image Pairs → Feature Detection → Feature Matching → Geometric Verification
```

**For LoFTR:**
- Direct dense matching without keypoint detection
- Coarse matching at 1/8 resolution
- Fine refinement at original resolution

**For SuperGlue:**
- SuperPoint keypoint detection
- Graph neural network matching
- Sinkhorn algorithm for optimal transport

### 4. 3D Reconstruction

```python
Matches → COLMAP Database → Incremental SfM → 3D Model
```

**COLMAP Pipeline:**
1. Import features and matches
2. Geometric verification (RANSAC)
3. Initialize with two views
4. Incrementally add images
5. Bundle adjustment optimization

## Key Design Decisions

### 1. Hybrid Matching Strategy

```python
if num_pairs > threshold:
    use_superglue()  # Faster for large scenes
else:
    use_loftr() + use_superglue()  # More accurate
```

**Rationale:** 
- Large scenes need speed → SuperGlue
- Small scenes need accuracy → LoFTR
- Combine both for best results

### 2. Multi-Scale Processing

```python
scales = [640, 1024, 1440]
for scale in scales:
    matches = match_at_scale(scale)
combine_and_deduplicate(matches)
```

**Benefits:**
- Small scale: Global structure
- Large scale: Fine details
- Combined: Best of both

### 3. Test-Time Augmentation (TTA)

```python
augmentations = ['original', 'flip_horizontal', 'rotate_90']
for aug in augmentations:
    matches = match_with_augmentation(aug)
combine_matches(matches)
```

**Why TTA helps:**
- More robust to viewpoint changes
- Finds matches missed in original orientation
- Minimal computational overhead

## Extension Points

### 1. Adding New Matchers

To add a new matching algorithm:

```python
class MyMatcher(BaseMatcher):
    def _setup(self):
        # Initialize your model
        pass
    
    def match_pair(self, img1_path, img2_path):
        # Implement matching logic
        return keypoints1, keypoints2, confidence
```

### 2. Custom Image Pair Selection

```python
class CustomPairSelector(ImagePairSelector):
    def select_pairs(self, images):
        # Implement custom logic
        return selected_pairs
```

### 3. Alternative Reconstruction Backends

```python
class OpenMVGInterface(ReconstructionInterface):
    def reconstruct(self, database_path):
        # Use OpenMVG instead of COLMAP
        pass
```

## Performance Considerations

### 1. GPU Memory Management

- Batch processing when possible
- Dynamic batch size based on available memory
- Gradient checkpointing for large models

### 2. Caching Strategy

- Cache global descriptors
- Cache preprocessed images
- Reuse COLMAP database when possible

### 3. Parallelization

- Parallel image loading
- Concurrent feature extraction
- Multi-threaded COLMAP operations

## Error Handling

### 1. Graceful Degradation

```python
try:
    matches = loftr_matcher.match()
except CUDAOutOfMemory:
    # Fall back to smaller resolution
    matches = loftr_matcher.match(resize=640)
```

### 2. Validation Checks

- Minimum number of images
- Image format validation
- Match count thresholds
- Geometric consistency checks

## Testing Strategy

### 1. Unit Tests

- Test each component independently
- Mock external dependencies
- Test edge cases

### 2. Integration Tests

- Test full pipeline with sample data
- Verify output format
- Check reconstruction metrics

### 3. Performance Tests

- Benchmark matching speed
- Memory usage profiling
- Scalability testing

## Future Enhancements

### 1. Real-time Reconstruction

- Streaming image processing
- Incremental matching
- Live visualization

### 2. Deep Learning Improvements

- Self-supervised training
- Domain adaptation
- Uncertainty estimation

### 3. Cloud Deployment

- Distributed processing
- API service
- Web interface

## Conclusion

Vision3D's architecture is designed for:
- **Modularity**: Easy to extend and modify
- **Performance**: Optimized for speed and accuracy
- **Usability**: Simple API hiding complexity
- **Robustness**: Handles edge cases gracefully

The clean separation of concerns makes it easy to understand, modify, and extend the system for your specific needs.