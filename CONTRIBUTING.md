# Contributing to OpenEmbeddings

Thank you for your interest in contributing to OpenEmbeddings! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that ensures a welcoming environment for all contributors. By participating, you agree to uphold this standard.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a feature branch
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/OpenEmbeddings.git
cd OpenEmbeddings

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# For FAISS acceleration
pip install faiss-cpu  # or faiss-gpu for GPU support

# For model quantization
pip install bitsandbytes

# For ONNX optimization
pip install onnxruntime

# For evaluation
pip install mteb
```

## Contributing Process

### 1. Choose an Issue

- Look for issues labeled `good first issue` for beginners
- Check existing issues before creating new ones
- Comment on issues you'd like to work on

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Follow the coding standards below
- Write tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Your Changes

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_models.py -v

# Run with coverage
python -m pytest tests/ --cov=openembeddings --cov-report=html
```

### 5. Update Documentation

- Update docstrings for new/modified functions
- Add examples for new features
- Update README.md if needed
- Update CHANGELOG.md

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters (not 79)
- Use double quotes for strings
- Use f-strings for formatting when possible

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black openembeddings/ tests/

# Sort imports
isort openembeddings/ tests/

# Type checking
mypy openembeddings/
```

### Type Hints

- All public functions must have type hints
- Use `from __future__ import annotations` for forward references
- Use `Optional[T]` for nullable types
- Use `Union[T, U]` for union types (Python < 3.10)

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.
    
    Longer description if needed. Explain the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 0.
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Testing

### Test Structure

- Unit tests in `tests/test_*.py`
- Integration tests for complex workflows
- Mock external dependencies when possible
- Test both success and failure cases

### Writing Tests

```python
import pytest
from openembeddings.models import DenseEmbedder

class TestDenseEmbedder:
    def test_initialization(self):
        """Test that DenseEmbedder initializes correctly."""
        embedder = DenseEmbedder()
        assert embedder is not None
        
    def test_embed_documents(self):
        """Test document embedding functionality."""
        embedder = DenseEmbedder()
        docs = ["test document", "another document"]
        embeddings = embedder.embed_documents(docs)
        
        assert len(embeddings) == len(docs)
        assert all(len(emb) > 0 for emb in embeddings)
        
    def test_invalid_input(self):
        """Test handling of invalid input."""
        embedder = DenseEmbedder()
        with pytest.raises(ValueError):
            embedder.embed_documents([])
```

### Test Coverage

- Aim for >90% test coverage
- Cover edge cases and error conditions
- Test with different model types and configurations

## Documentation

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Document configuration options

### User Documentation

- Update README.md for new features
- Add examples to `examples/` directory
- Update CLI help text
- Create tutorials for complex features

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] Type checking passes (mypy)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated

### PR Description Template

```markdown
## Description

Brief description of the changes.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update

## Testing

- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Tested with different model types

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Automated checks (CI/CD) must pass
2. At least one maintainer review required
3. Address review feedback promptly
4. Squash commits before merging (if requested)

## Issue Reporting

### Bug Reports

Include:

- Python version and OS
- OpenEmbeddings version
- Minimal code to reproduce
- Expected vs actual behavior
- Error messages and stack traces

### Feature Requests

Include:

- Clear description of the feature
- Use case and motivation
- Proposed API (if applicable)
- Willingness to implement

### Performance Issues

Include:

- Benchmark code
- Performance measurements
- System specifications
- Profiling data (if available)

## Development Guidelines

### Architecture Principles

- Modular design with clear interfaces
- Dependency injection for testability
- Graceful error handling and fallbacks
- Performance-conscious implementations
- Memory-efficient operations

### Adding New Models

1. Create model class in `openembeddings/models/`
2. Implement required interface methods
3. Add comprehensive tests
4. Update CLI integration
5. Add documentation and examples

### Adding New Features

1. Design API carefully (breaking changes are costly)
2. Consider backward compatibility
3. Add configuration options where appropriate
4. Implement proper error handling
5. Add monitoring/logging hooks

## Community

### Getting Help

- GitHub Discussions for questions
- GitHub Issues for bugs and features
- Email: nikjois@llamasearch.ai for urgent matters

### Recognition

Contributors are recognized in:

- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to OpenEmbeddings! Your contributions help make this project better for everyone. 