"""
I/O utilities for Vision3D.

This module provides functions for loading and saving various data formats
used in 3D reconstruction pipelines.
"""

import json
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
import pickle

logger = logging.getLogger(__name__)


def load_image(
    image_path: Union[str, Path],
    grayscale: bool = False,
    resize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load an image from disk with optional preprocessing.
    
    Args:
        image_path: Path to the image file
        grayscale: Convert to grayscale if True
        resize: Resize to (width, height) if provided
    
    Returns:
        Loaded image as numpy array
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    if grayscale:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(image_path))
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Resize if requested
    if resize is not None:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)
    
    return img


def save_results(
    results: Dict,
    output_path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save reconstruction results to disk.
    
    Args:
        results: Results dictionary to save
        output_path: Output file path
        format: Output format ('json', 'pickle', 'h5')
    
    Raises:
        ValueError: If format is not supported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = _prepare_for_json(results)
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif format == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
    
    elif format == 'h5':
        _save_to_h5(results, output_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results saved to {output_path}")


def load_results(
    input_path: Union[str, Path],
    format: str = 'json'
) -> Dict:
    """
    Load reconstruction results from disk.
    
    Args:
        input_path: Input file path
        format: Input format ('json', 'pickle', 'h5')
    
    Returns:
        Loaded results dictionary
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If format is not supported
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    if format == 'json':
        with open(input_path, 'r') as f:
            results = json.load(f)
        # Convert lists back to numpy arrays
        results = _restore_from_json(results)
    
    elif format == 'pickle':
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
    
    elif format == 'h5':
        results = _load_from_h5(input_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Results loaded from {input_path}")
    return results


def save_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_path: Union[str, Path],
    format: str = 'ply'
) -> None:
    """
    Save 3D point cloud to file.
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-255)
        output_path: Output file path
        format: Output format ('ply', 'xyz', 'pcd')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'ply':
        _save_ply(points, colors, output_path)
    elif format == 'xyz':
        _save_xyz(points, colors, output_path)
    elif format == 'pcd':
        _save_pcd(points, colors, output_path)
    else:
        raise ValueError(f"Unsupported point cloud format: {format}")
    
    logger.info(f"Point cloud saved to {output_path}")


def _save_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_path: Path
) -> None:
    """Save point cloud in PLY format."""
    num_points = len(points)
    
    # PLY header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]
    
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    
    header.append("end_header")
    
    # Write file
    with open(output_path, 'w') as f:
        # Write header
        for line in header:
            f.write(line + '\n')
        
        # Write points
        for i in range(num_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            
            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            
            f.write(line + '\n')


def _save_xyz(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_path: Path
) -> None:
    """Save point cloud in XYZ format."""
    if colors is not None:
        data = np.hstack([points, colors])
        fmt = '%.6f %.6f %.6f %d %d %d'
    else:
        data = points
        fmt = '%.6f %.6f %.6f'
    
    np.savetxt(output_path, data, fmt=fmt)


def _save_pcd(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    output_path: Path
) -> None:
    """Save point cloud in PCD format."""
    # This would implement PCD format saving
    # For now, we'll use a simple text format
    raise NotImplementedError("PCD format not yet implemented")


def _prepare_for_json(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _prepare_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_prepare_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def _restore_from_json(obj):
    """Recursively convert lists back to numpy arrays."""
    if isinstance(obj, dict):
        # Check if this should be a numpy array (heuristic)
        if 'data' in obj and isinstance(obj['data'], list):
            return np.array(obj['data'])
        else:
            return {k: _restore_from_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Check if this is a numeric array
        if obj and all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj)
        else:
            return [_restore_from_json(v) for v in obj]
    else:
        return obj


def _save_to_h5(data: Dict, output_path: Path) -> None:
    """Save data to HDF5 format."""
    with h5py.File(output_path, 'w') as f:
        _write_h5_recursive(f, '', data)


def _write_h5_recursive(f, path: str, data):
    """Recursively write data to HDF5."""
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}/{key}" if path else key
            _write_h5_recursive(f, new_path, value)
    elif isinstance(data, (np.ndarray, list)):
        f.create_dataset(path, data=np.array(data))
    else:
        # Store as attribute for simple types
        parts = path.rsplit('/', 1)
        if len(parts) == 2:
            group_path, attr_name = parts
            if group_path not in f:
                f.create_group(group_path)
            f[group_path].attrs[attr_name] = data
        else:
            f.attrs[path] = data


def _load_from_h5(input_path: Path) -> Dict:
    """Load data from HDF5 format."""
    data = {}
    with h5py.File(input_path, 'r') as f:
        _read_h5_recursive(f, data)
    return data


def _read_h5_recursive(obj, data: Dict, path: str = ''):
    """Recursively read data from HDF5."""
    if isinstance(obj, h5py.Group):
        for key in obj.keys():
            new_path = f"{path}.{key}" if path else key
            if isinstance(obj[key], h5py.Dataset):
                _set_nested_dict(data, new_path, obj[key][:])
            else:
                _read_h5_recursive(obj[key], data, new_path)
        
        # Read attributes
        for key, value in obj.attrs.items():
            attr_path = f"{path}.{key}" if path else key
            _set_nested_dict(data, attr_path, value)


def _set_nested_dict(d: Dict, path: str, value):
    """Set value in nested dictionary using dot notation."""
    parts = path.split('.')
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value