# Contributing to Vision3D

First off, thank you for considering contributing to Vision3D! It's people like you that make Vision3D such a great tool for the computer vision community.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, images, etc.)
- **Describe the behavior you observed and expected**
- **Include system information** (OS, Python version, GPU model, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding standards** (see below)
3. **Write tests** for your changes
4. **Ensure all tests pass**
5. **Update documentation** as needed
6. **Submit a pull request**

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/yourusername/vision3d.git
   cd vision3d
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:
- Line length: 88 characters (Black default)
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Code Formatting

We use the following tools (automatically run via pre-commit):
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run manually:
```bash
black vision3d/
isort vision3d/
flake8 vision3d/
mypy vision3d/
```

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Provide examples in docstrings when helpful

Example:
```python
def match_features(
    image1: np.ndarray,
    image2: np.ndarray,
    method: str = "loftr"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match features between two images.
    
    Args:
        image1: First image as numpy array (H, W, 3)
        image2: Second image as numpy array (H, W, 3)
        method: Matching method to use ('loftr' or 'superglue')
    
    Returns:
        Tuple containing:
            - keypoints1: Nx2 array of keypoints in image1
            - keypoints2: Nx2 array of keypoints in image2
            - confidence: Nx1 array of match confidences
    
    Example:
        >>> kpts1, kpts2, conf = match_features(img1, img2, method='loftr')
        >>> print(f"Found {len(kpts1)} matches")
    """
    # Implementation
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vision3d

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Include both positive and negative test cases
- Mock external dependencies when appropriate

Example test:
```python
def test_loftr_matcher_initialization():
    """Test LoFTR matcher initialization."""
    matcher = LoFTRMatcher(device='cpu')
    assert matcher.device == torch.device('cpu')
    assert matcher.config['confidence_threshold'] == 0.3
```

## Project Structure

```
vision3d/
├── core/           # Core functionality
├── models/         # Model implementations
├── utils/          # Utility functions
├── data/           # Data handling
├── scripts/        # CLI scripts
└── tests/          # Test files
```

### Adding New Features

1. **New Matcher**: Inherit from `BaseMatcher` in `models/base.py`
2. **New Utility**: Add to appropriate module in `utils/`
3. **New Pipeline**: Extend `Vision3DPipeline` in `core/pipeline.py`

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Writing Documentation

- Update docstrings when changing functionality
- Add examples for new features
- Update README.md for user-facing changes
- Create tutorials for complex features

## Release Process

1. Update version in `setup.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create a pull request
4. After merge, tag the release:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

## Questions?

Feel free to:
- Open an issue for questions
- Join our discussions
- Contact the maintainers

Thank you for contributing to Vision3D! 🚀