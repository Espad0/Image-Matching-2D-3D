#!/usr/bin/env python3
"""
Simple 3D reconstruction viewer using COLMAP GUI and export tools
"""

import subprocess
import os
from pathlib import Path


def view_with_colmap_gui(model_path):
    """Open COLMAP GUI to view the reconstruction"""
    print("Opening COLMAP GUI...")
    print("In the GUI:")
    print("1. Go to 'File' -> 'Import model'")
    print("2. Select the folder:", model_path)
    print("3. The 3D reconstruction will be displayed")
    
    try:
        subprocess.run(["colmap", "gui"], check=True)
    except subprocess.CalledProcessError:
        print("\nCOLMAP GUI not available. Try alternative methods below.")
    except FileNotFoundError:
        print("\nCOLMAP GUI not found. Make sure COLMAP is installed and in PATH.")


def export_to_text(model_path, output_dir):
    """Export COLMAP model to text format for easier viewing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nExporting to text format in {output_dir}...")
    
    try:
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", model_path,
            "--output_path", output_dir,
            "--output_type", "TXT"
        ], check=True)
        print("Text files created:")
        print(f"  - {output_dir}/points3D.txt")
        print(f"  - {output_dir}/images.txt")
        print(f"  - {output_dir}/cameras.txt")
    except Exception as e:
        print(f"Export failed: {e}")


def export_to_ply(model_path, output_path):
    """Export to PLY using COLMAP"""
    print(f"\nExporting to PLY format: {output_path}")
    
    try:
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", model_path,
            "--output_path", output_path,
            "--output_type", "PLY"
        ], check=True)
        print(f"PLY file created: {output_path}")
        print("\nYou can view this file in:")
        print("  - MeshLab: https://www.meshlab.net/")
        print("  - CloudCompare: https://www.cloudcompare.org/")
        print("  - Blender: https://www.blender.org/")
        print("  - Online: https://www.3dviewer.net/")
    except Exception as e:
        print(f"PLY export failed: {e}")


def create_simple_viewer():
    """Create a simple web-based viewer using three.js"""
    viewer_html = '''<!DOCTYPE html>
<html>
<head>
    <title>3D Point Cloud Viewer</title>
    <style>
        body { margin: 0; overflow: hidden; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            font-family: Arial;
        }
    </style>
</head>
<body>
    <div id="info">
        <h3>3D Reconstruction Viewer</h3>
        <p>Drag PLY file here or use File > Open</p>
        <p>Controls: Left mouse - rotate, Right mouse - pan, Scroll - zoom</p>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.150.0/examples/js/loaders/PLYLoader.js"></script>
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x222222);
        
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(5, 5, 5);
        
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        
        // Lighting
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        scene.add(light);
        scene.add(new THREE.AmbientLight(0x404040));
        
        // Grid
        const gridHelper = new THREE.GridHelper(10, 10);
        scene.add(gridHelper);
        
        // PLY Loader
        const loader = new THREE.PLYLoader();
        
        // File drop
        renderer.domElement.addEventListener('drop', (e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file && file.name.endsWith('.ply')) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    const geometry = loader.parse(event.target.result);
                    geometry.computeVertexNormals();
                    
                    const material = new THREE.PointsMaterial({
                        size: 0.01,
                        vertexColors: true
                    });
                    
                    const points = new THREE.Points(geometry, material);
                    scene.add(points);
                    
                    // Center camera on points
                    geometry.computeBoundingBox();
                    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
                    controls.target.copy(center);
                };
                reader.readAsArrayBuffer(file);
            }
        });
        
        renderer.domElement.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>'''
    
    with open('viewer.html', 'w') as f:
        f.write(viewer_html)
    
    print("\nCreated viewer.html - Open this file in a web browser and drag your PLY file onto it")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View 3D reconstruction results")
    parser.add_argument("--model-path", type=str, default="output_multires/sparse/0",
                        help="Path to COLMAP reconstruction")
    args = parser.parse_args()
    
    print("3D Reconstruction Viewing Options:")
    print("=" * 50)
    
    # Option 1: COLMAP GUI
    print("\n1. COLMAP GUI (if installed):")
    view_with_colmap_gui(args.model_path)
    
    # Option 2: Export to text
    print("\n2. Export to text format:")
    export_to_text(args.model_path, "output_text")
    
    # Option 3: Export to PLY
    print("\n3. Export to PLY:")
    export_to_ply(args.model_path, "reconstruction.ply")
    
    # Option 4: Web viewer
    print("\n4. Web-based viewer:")
    create_simple_viewer()
    
    print("\n" + "=" * 50)
    print("Choose the option that works best for your setup!")