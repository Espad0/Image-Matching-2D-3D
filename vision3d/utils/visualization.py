"""
Visualization utilities for Vision3D.

This module provides functions for visualizing:
- Feature matches between images
- 3D reconstructions
- Point clouds
- Camera poses
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def visualize_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    max_matches: int = 500,
    line_thickness: int = 1,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize feature matches between two images.
    
    Args:
        img1: First image (BGR)
        img2: Second image (BGR)
        kpts1: Keypoints in first image (Nx2)
        kpts2: Keypoints in second image (Nx2)
        confidence: Match confidence scores (optional)
        max_matches: Maximum number of matches to display
        line_thickness: Thickness of match lines
        save_path: Path to save visualization
    
    Returns:
        Visualization image
    """
    # Limit number of matches for clarity
    if len(kpts1) > max_matches:
        if confidence is not None:
            # Select top matches by confidence
            indices = np.argsort(confidence)[-max_matches:]
            kpts1 = kpts1[indices]
            kpts2 = kpts2[indices]
            confidence = confidence[indices]
        else:
            # Random selection
            indices = np.random.choice(len(kpts1), max_matches, replace=False)
            kpts1 = kpts1[indices]
            kpts2 = kpts2[indices]
    
    # Create side-by-side visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:] = img2
    
    # Generate colors based on confidence
    if confidence is not None:
        # Normalize confidence to [0, 1]
        conf_norm = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-6)
        colors = cm.RdYlGn(conf_norm)[:, :3] * 255
    else:
        # Random colors
        colors = np.random.randint(0, 255, (len(kpts1), 3))
    
    # Draw matches
    for i, (pt1, pt2, color) in enumerate(zip(kpts1, kpts2, colors)):
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple((pt2 + [w1, 0]).astype(int))
        color = tuple(int(c) for c in color)
        
        # Draw line
        cv2.line(vis, pt1, pt2, color, line_thickness)
        
        # Draw keypoints
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)
    
    # Add text information
    text = f"{len(kpts1)} matches"
    if confidence is not None:
        text += f" (avg conf: {np.mean(confidence):.3f})"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, vis)
        logger.info(f"Saved match visualization to {save_path}")
    
    return vis


def visualize_reconstruction(
    reconstruction: Dict,
    save_path: Optional[str] = None,
    show_cameras: bool = True,
    show_points: bool = True,
    point_size: int = 2,
    camera_size: float = 0.1
) -> Optional[go.Figure]:
    """
    Visualize 3D reconstruction using Plotly.
    
    Args:
        reconstruction: Reconstruction dictionary
        save_path: Path to save HTML visualization
        show_cameras: Whether to show camera positions
        show_points: Whether to show 3D points
        point_size: Size of 3D points
        camera_size: Size of camera markers
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add 3D points
    if show_points and reconstruction['points3D']:
        points = np.array([p['xyz'] for p in reconstruction['points3D'].values()])
        colors = np.array([p['rgb'] for p in reconstruction['points3D'].values()])
        
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors,
                colorscale='RGB',
                opacity=0.8,
            ),
            name='3D Points'
        ))
    
    # Add camera positions
    if show_cameras and reconstruction['images']:
        camera_positions = []
        camera_orientations = []
        
        for img_data in reconstruction['images'].values():
            # Camera position is -R^T * t
            R = np.array(img_data['R'])
            t = np.array(img_data['t'])
            pos = -R.T @ t
            camera_positions.append(pos)
            
            # Camera viewing direction
            direction = R.T @ np.array([0, 0, 1])
            camera_orientations.append(direction)
        
        camera_positions = np.array(camera_positions)
        
        # Add camera markers
        fig.add_trace(go.Scatter3d(
            x=camera_positions[:, 0],
            y=camera_positions[:, 1],
            z=camera_positions[:, 2],
            mode='markers',
            marker=dict(
                size=camera_size * 100,
                color='red',
                symbol='diamond',
            ),
            name='Cameras'
        ))
        
        # Add camera viewing directions
        for pos, direction in zip(camera_positions, camera_orientations):
            end_point = pos + camera_size * direction
            fig.add_trace(go.Scatter3d(
                x=[pos[0], end_point[0]],
                y=[pos[1], end_point[1]],
                z=[pos[2], end_point[2]],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title=f"3D Reconstruction ({reconstruction['num_points3D']} points, {reconstruction['num_images']} cameras)",
        showlegend=True
    )
    
    # Save if requested
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved 3D visualization to {save_path}")
    
    return fig


def plot_reconstruction_statistics(
    reconstruction: Dict,
    save_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Plot various statistics about the reconstruction.
    
    Args:
        reconstruction: Reconstruction dictionary
        save_dir: Directory to save plots
    
    Returns:
        Dictionary of matplotlib figures
    """
    figures = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Reprojection error distribution
    if reconstruction['points3D']:
        errors = [p['error'] for p in reconstruction['points3D'].values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Number of Points')
        ax.set_title('Distribution of Reprojection Errors')
        ax.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(errors):.2f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', 
                   label=f'Median: {np.median(errors):.2f}')
        ax.legend()
        
        figures['reprojection_errors'] = fig
        if save_dir:
            fig.savefig(save_dir / 'reprojection_errors.png', dpi=150, bbox_inches='tight')
    
    # 2. Track length distribution
    if reconstruction['points3D']:
        track_lengths = [len(p['image_ids']) for p in reconstruction['points3D'].values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(track_lengths, bins=range(2, max(track_lengths) + 2), 
                alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Track Length (number of images)')
        ax.set_ylabel('Number of Points')
        ax.set_title('Distribution of Track Lengths')
        ax.axvline(np.mean(track_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(track_lengths):.1f}')
        ax.legend()
        
        figures['track_lengths'] = fig
        if save_dir:
            fig.savefig(save_dir / 'track_lengths.png', dpi=150, bbox_inches='tight')
    
    # 3. Camera distribution
    if reconstruction['images']:
        # Extract camera positions
        positions = []
        for img_data in reconstruction['images'].values():
            R = np.array(img_data['R'])
            t = np.array(img_data['t'])
            pos = -R.T @ t
            positions.append(pos)
        positions = np.array(positions)
        
        # Plot camera positions from above (X-Y plane)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(positions[:, 0], positions[:, 1], c='red', s=100, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Camera Positions (Top View)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        figures['camera_positions'] = fig
        if save_dir:
            fig.savefig(save_dir / 'camera_positions.png', dpi=150, bbox_inches='tight')
    
    # 4. Summary statistics
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reconstruction Summary', fontsize=16)
    
    # Registered vs total images
    total_images = reconstruction.get('total_images', len(reconstruction['images']))
    registered = reconstruction['num_images']
    ax1.bar(['Total', 'Registered'], [total_images, registered], 
            color=['lightblue', 'darkblue'])
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Image Registration')
    
    # Point cloud size
    ax2.text(0.5, 0.5, f"{reconstruction['num_points3D']:,}\n3D Points", 
             ha='center', va='center', fontsize=24, transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Point Cloud Size')
    
    # Mean metrics
    if reconstruction['statistics']:
        stats = reconstruction['statistics']
        metrics = ['Reproj Error', 'Track Length']
        values = [stats['mean_reprojection_error'], stats['mean_track_length']]
        ax3.bar(metrics, values, color=['orange', 'green'])
        ax3.set_ylabel('Value')
        ax3.set_title('Mean Metrics')
    
    # Camera models
    if reconstruction['cameras']:
        models = [cam['model'] for cam in reconstruction['cameras'].values()]
        from collections import Counter
        model_counts = Counter(models)
        ax4.pie(model_counts.values(), labels=model_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Camera Models')
    
    figures['summary'] = fig
    if save_dir:
        fig.savefig(save_dir / 'summary.png', dpi=150, bbox_inches='tight')
    
    plt.close('all')  # Close all figures to free memory
    
    return figures


def create_match_matrix(
    image_paths: List[str],
    matches: Dict[Tuple[int, int], Dict],
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a matrix visualization of matches between all image pairs.
    
    Args:
        image_paths: List of image paths
        matches: Dictionary of matches with (idx1, idx2) keys
        save_path: Path to save visualization
    
    Returns:
        Match matrix
    """
    n_images = len(image_paths)
    matrix = np.zeros((n_images, n_images))
    
    # Fill matrix with number of matches
    for (i, j), match_data in matches.items():
        num_matches = match_data.get('num_matches', len(match_data.get('keypoints1', [])))
        matrix[i, j] = num_matches
        matrix[j, i] = num_matches  # Symmetric
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Matches', rotation=270, labelpad=20)
    
    # Set ticks and labels
    image_names = [Path(p).name for p in image_paths]
    ax.set_xticks(range(n_images))
    ax.set_yticks(range(n_images))
    ax.set_xticklabels(image_names, rotation=45, ha='right')
    ax.set_yticklabels(image_names)
    
    # Add text annotations
    for i in range(n_images):
        for j in range(n_images):
            if i != j and matrix[i, j] > 0:
                text = ax.text(j, i, f'{int(matrix[i, j])}',
                             ha='center', va='center', color='black' if matrix[i, j] < matrix.max()/2 else 'white')
    
    ax.set_title('Feature Matches Between Image Pairs')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return matrix