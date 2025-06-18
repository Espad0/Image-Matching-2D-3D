"""Setup script for Vision3D package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
with open("requirements.txt") as f:
    requirements = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]

setup(
    name="vision3d",
    version="1.0.0",
    author="Andrej Nesterov",
    author_email="your.email@example.com",
    description="State-of-the-art 3D reconstruction from images using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vision3d",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/vision3d/issues",
        "Documentation": "https://vision3d.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/vision3d",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "notebook": [
            "notebook>=6.4.0",
            "ipywidgets>=7.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vision3d=vision3d.cli:main",
            "vision3d-reconstruct=vision3d.scripts.reconstruct:main",
            "vision3d-benchmark=vision3d.scripts.benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vision3d": ["data/*.yaml", "configs/*.yaml"],
    },
    zip_safe=False,
)