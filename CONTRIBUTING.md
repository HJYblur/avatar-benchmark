# Contributing to Avatar Benchmark

Thank you for your interest in contributing to the Avatar Benchmark project! This document provides guidelines for contributing to the project.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/avatar-benchmark.git
   cd avatar-benchmark
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run Tests**
   ```bash
   python scripts/test_structure.py
   ```

## Project Structure

```
avatar-benchmark/
├── configs/           # Configuration files
├── data/             # Input data directory
├── models/           # Model files (SMPL-X, PyMAF-X)
├── outputs/          # Pipeline outputs
├── src/              # Source code
│   ├── preprocessing/  # Video preprocessing
│   ├── inference/      # PyMAF-X inference
│   ├── rendering/      # Mesh generation & rendering
│   ├── metrics/        # Evaluation metrics
│   └── utils/          # Utility functions
└── scripts/          # Main pipeline scripts
```

## Adding New Features

### Adding a New Metric

1. Add your metric function to `src/metrics/evaluator.py`:
   ```python
   def calculate_new_metric(self, img1: np.ndarray, img2: np.ndarray) -> float:
       """Calculate new metric."""
       # Your implementation
       return metric_value
   ```

2. Update the `evaluate_sequence` method to include your metric

3. Update `configs/default.yaml` to add configuration options

### Adding a New Rendering Method

1. Add your method to `src/rendering/renderer.py`:
   ```python
   def render_with_new_method(self, mesh: trimesh.Trimesh) -> np.ndarray:
       """New rendering method."""
       # Your implementation
       return rendered_image
   ```

2. Update configuration as needed

### Adding a New Preprocessing Step

1. Add your preprocessor to `src/preprocessing/`:
   ```python
   class NewPreprocessor:
       def process(self, data):
           # Your implementation
           return processed_data
   ```

2. Integrate with the main pipeline in `scripts/run_pipeline.py`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Example Code Style

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Implementation
    return True
```

## Testing

Before submitting a pull request:

1. **Verify Structure**
   ```bash
   python scripts/test_structure.py
   ```

2. **Test Your Changes**
   - Run the pipeline with test data
   - Verify outputs are correct
   - Check that no existing functionality is broken

3. **Document Your Changes**
   - Update README if adding new features
   - Add/update docstrings
   - Create examples if applicable

## Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, documented code
   - Follow the project structure
   - Add tests if applicable

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

4. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Submit Pull Request**
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Describe your changes in detail
   - Reference any related issues

## Commit Message Format

Use clear, descriptive commit messages:

- `Add: new feature or file`
- `Fix: bug fix`
- `Update: changes to existing feature`
- `Refactor: code restructuring`
- `Docs: documentation changes`

Examples:
```
Add: LPIPS metric calculation
Fix: mesh rendering orientation issue
Update: improve frame extraction efficiency
Docs: add pipeline architecture diagram
```

## Adding Dependencies

When adding new dependencies:

1. Add to `requirements.txt`
2. Update `setup.py` if needed
3. Document why the dependency is needed
4. Check for version compatibility

## Documentation

Good documentation includes:

1. **Code Comments**
   - Explain complex logic
   - Document assumptions
   - Note any limitations

2. **Docstrings**
   - All public functions/classes
   - Include Args, Returns, Raises

3. **README Updates**
   - New features
   - Changed behavior
   - New dependencies

## Questions and Support

- Open an issue for bugs or feature requests
- Use discussions for questions
- Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be acknowledged in the project README and release notes.

Thank you for contributing! 🎉
