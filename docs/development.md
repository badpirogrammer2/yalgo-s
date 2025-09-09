# YALGO-S Development Guide

## Overview

This guide covers development practices, contribution guidelines, and technical details for YALGO-S contributors.

## 🏗️ Development Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Git
git --version

# CUDA (optional, for GPU development)
nvcc --version
```

### Clone and Setup
```bash
# Clone repository
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"
```

### Development Tools
```bash
# Install pre-commit hooks
pre-commit install

# Run initial checks
pre-commit run --all-files
```

## 🧪 Testing Framework

### Running Tests
```bash
# Run all tests
python run_all_tests.py

# Run specific test suites
python test_agmohd_hf.py
python test_poic_net_hf.py
python test_arce_hf.py

# Run with coverage
pytest --cov=yalgo_s --cov-report=html

# Run GPU tests
python -m pytest tests/ -k gpu -v
```

### Writing Tests
```python
import pytest
import torch
from yalgo_s import AGMOHD

def test_agmohd_basic():
    """Test basic AGMOHD functionality."""
    model = torch.nn.Linear(10, 1)
    optimizer = AGMOHD(model, lr=0.01)

    # Test optimizer creation
    assert optimizer is not None
    assert optimizer.lr == 0.01

def test_agmohd_training():
    """Test AGMOHD training loop."""
    model = torch.nn.Linear(10, 1)
    optimizer = AGMOHD(model)

    # Create dummy data
    X = torch.randn(100, 10)
    y = X.sum(dim=1, keepdim=True)

    # Training loop
    for _ in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()

    # Verify training worked
    assert loss.item() < 1.0
```

### Test Organization
```
tests/
├── test_agmohd.py          # AGMOHD tests
├── test_poic_net.py        # POIC-NET tests
├── test_arce.py           # ARCE tests
├── test_image_training.py # Image training tests
├── test_integration.py    # Integration tests
├── test_performance.py    # Performance tests
└── conftest.py           # Test configuration
```

## 📝 Code Style and Standards

### Python Style Guide
```python
# Use type hints
def train_model(model: nn.Module, data: DataLoader) -> Dict[str, float]:
    """Train model and return metrics."""
    pass

# Use descriptive variable names
learning_rate = 0.01  # Not lr
batch_size = 32        # Not bs

# Use docstrings
class AGMOHD:
    """Adaptive Gradient Momentum with Hindrance Detection optimizer.

    This optimizer dynamically adjusts learning rates and momentum
    based on gradient analysis and training hindrance detection.
    """
    pass
```

### Code Formatting
```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check style with flake8
flake8 .

# Type checking with mypy
mypy .
```

### Commit Messages
```bash
# Good commit messages
git commit -m "feat: add RTX 5060 optimizations to AGMOHD"

git commit -m "fix: resolve memory leak in POIC-NET inference"

git commit -m "docs: update installation guide for macOS"

git commit -m "test: add GPU memory tests for ImageTrainer"

# Bad commit messages
git commit -m "fixed bug"
git commit -m "updated code"
git commit -m "changes"
```

## 🏗️ Architecture Guidelines

### Module Structure
```
yalgo_s/
├── __init__.py           # Main exports
├── agmohd/              # AGMOHD optimizer
│   ├── __init__.py
│   ├── agmohd.py       # Core implementation
│   └── utils.py        # Helper functions
├── poic_net/           # POIC-NET algorithm
│   ├── __init__.py
│   ├── poicnet.py     # Core implementation
│   └── models.py      # Model definitions
├── arce/               # ARCE algorithm
│   ├── __init__.py
│   ├── arce.py        # Core implementation
│   └── context.py     # Context handling
├── image_training/     # Image training module
│   ├── __init__.py
│   ├── trainer.py     # Training logic
│   ├── data.py        # Data loading
│   └── models.py      # Model architectures
└── utils/              # Shared utilities
    ├── __init__.py
    ├── gpu.py         # GPU utilities
    ├── logging.py     # Logging utilities
    └── metrics.py     # Performance metrics
```

### Class Design
```python
class BaseOptimizer:
    """Base class for YALGO-S optimizers."""

    def __init__(self, model, lr=0.01, **kwargs):
        self.model = model
        self.lr = lr
        self.device = kwargs.get('device', 'auto')
        self._setup_device()

    def _setup_device(self):
        """Setup device (CPU/GPU) configuration."""
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def step(self):
        """Perform optimization step."""
        raise NotImplementedError

    def zero_grad(self):
        """Zero gradients."""
        self.model.zero_grad()
```

### Error Handling
```python
class YALGOSError(Exception):
    """Base exception for YALGO-S errors."""
    pass

class GPUError(YALGOSError):
    """GPU-related errors."""
    pass

class MemoryError(YALGOSError):
    """Memory-related errors."""
    pass

def safe_gpu_operation(func):
    """Decorator for safe GPU operations."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                raise MemoryError("GPU out of memory") from e
            elif "device" in str(e):
                raise GPUError("GPU device error") from e
            else:
                raise
    return wrapper
```

## 🚀 Performance Optimization

### GPU Optimization
```python
# Enable cuDNN optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Memory management
torch.cuda.set_per_process_memory_fraction(0.8)

# RTX-specific optimizations
if torch.cuda.get_device_name().startswith('RTX'):
    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

### Memory Optimization
```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

def create_checkpointed_model():
    """Create model with gradient checkpointing."""
    pass

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Profiling
```python
import torch.profiler

def profile_training():
    """Profile training performance."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        # Training code here
        pass

    print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📦 Release Process

### Version Management
```python
# Version format: MAJOR.MINOR.PATCH
__version__ = "0.1.0"

# Update version in:
# - yalgo_s/__init__.py
# - setup.py
# - pyproject.toml
# - docs/conf.py
```

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release branch
- [ ] Tag release
- [ ] Create GitHub release
- [ ] Update PyPI package

### Building Distributions
```bash
# Build source distribution
python setup.py sdist

# Build wheel
python setup.py bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## 🔧 CI/CD Pipeline

### GitHub Actions
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        python run_all_tests.py

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
```

## 📚 Documentation

### Building Docs
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

### Documentation Standards
```python
def function_with_docstring(param1: int, param2: str) -> bool:
    """Function that does something useful.

    This function performs a specific operation with detailed
    explanation of its behavior and parameters.

    Args:
        param1 (int): Description of first parameter
        param2 (str): Description of second parameter

    Returns:
        bool: Description of return value

    Raises:
        ValueError: When invalid parameters are provided
        RuntimeError: When operation fails

    Examples:
        >>> result = function_with_docstring(1, "test")
        >>> print(result)
        True
    """
    pass
```

## 🤝 Contributing Guidelines

### Issue Reporting
- Use issue templates
- Provide minimal reproducible examples
- Include system information
- Attach relevant logs

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Ensure all tests pass
7. Submit pull request

### Code Review Process
- All PRs require review
- Maintainers review within 2-3 business days
- Address review comments
- Squash commits before merge
- Use conventional commit messages

## 🔒 Security

### Security Best Practices
```python
# Avoid hardcoding secrets
# Use environment variables
import os
api_key = os.getenv('YALGO_S_API_KEY')

# Validate inputs
def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    # Additional validation logic

# Use secure random for tokens
import secrets
token = secrets.token_urlsafe(32)
```

### Vulnerability Reporting
- Report security issues privately
- Use GitHub Security tab
- Allow 90 days for fixes
- Responsible disclosure

## 📊 Performance Monitoring

### Benchmarking
```python
import time
from contextlib import contextmanager

@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Elapsed time: {end - start:.3f}s")

# Usage
with timer():
    result = expensive_operation()
```

### Metrics Collection
```python
class PerformanceMonitor:
    """Monitor performance metrics during training."""

    def __init__(self):
        self.metrics = []

    def log_metric(self, name, value, step):
        """Log a performance metric."""
        self.metrics.append({
            'name': name,
            'value': value,
            'step': step,
            'timestamp': time.time()
        })

    def get_summary(self):
        """Get performance summary."""
        return {
            'total_steps': len(self.metrics),
            'avg_values': self._calculate_averages(),
            'peak_memory': self._get_peak_memory()
        }
```

## 🎯 Best Practices

### Code Quality
- Write self-documenting code
- Use meaningful variable names
- Keep functions small and focused
- Add comprehensive error handling
- Write tests for all new features

### Performance
- Profile code before optimizing
- Use appropriate data structures
- Minimize memory allocations
- Leverage GPU acceleration when possible
- Cache expensive computations

### Maintainability
- Follow SOLID principles
- Use dependency injection
- Keep coupling low
- Document design decisions
- Refactor regularly

### Collaboration
- Communicate clearly
- Respect coding standards
- Help others learn
- Share knowledge
- Be open to feedback

This development guide ensures consistent, high-quality contributions to the YALGO-S project while maintaining performance and reliability standards.
