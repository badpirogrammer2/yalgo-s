# YALGO-S Installation Guide

## Overview

This guide provides comprehensive installation instructions for YALGO-S across different platforms and environments.

## Quick Start Installation

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos

# Install core dependencies
pip install -e .
```

### Full Installation
```bash
# Install with all optional dependencies
pip install -e ".[all]"
```

## Platform-Specific Installation

### Linux (Ubuntu/Debian)
```bash
# System dependencies
sudo apt update
sudo apt install -y python3-dev build-essential

# NVIDIA drivers (if using GPU)
sudo apt install -y nvidia-driver-470

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install YALGO-S
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos
pip install -e .
```

### macOS (Intel)
```bash
# Install Python (if not already installed)
brew install python@3.9

# Install PyTorch
pip install torch torchvision torchaudio

# Install YALGO-S
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos
pip install -e .
```

### macOS (Apple Silicon)
```bash
# Python comes pre-installed
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# MPS acceleration is automatically detected
# No additional setup required

# Install YALGO-S
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos
pip install -e .
```

### Windows
```bash
# Install Python from python.org or Microsoft Store
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install YALGO-S
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s\ALGOs\New Algos
pip install -e .
```

## GPU Setup

### NVIDIA RTX 5060 Setup
```bash
# Install CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Multi-GPU Setup
```bash
# Enable multi-GPU support
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Verify multi-GPU
python -c "import torch; print(torch.cuda.device_count())"
```

## Testing Installation

### Basic Tests
```bash
# Test core functionality
python -c "import yalgo_s; print('YALGO-S installed successfully!')"

# Run comprehensive test suite
python run_all_tests.py
```

### GPU Tests
```bash
# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test RTX optimizations
python test_parallel_optimizations.py
```

## Troubleshooting

### Common Issues

**CUDA Installation Problems**
```bash
# Check CUDA version compatibility
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Memory Issues**
```bash
# Reduce batch size
export CUDA_VISIBLE_DEVICES=0
python your_script.py --batch-size 16

# Enable memory optimization
python -c "import torch; torch.cuda.set_per_process_memory_fraction(0.8)"
```

**Import Errors**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip uninstall yalgo-s
pip install -e .
```

## Environment Configuration

### Environment Variables
```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1    # Specify GPU devices
export TORCH_USE_CUDA_DSA=1        # CUDA device-side assertions

# Memory Configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization

# Performance Configuration
export OMP_NUM_THREADS=8            # OpenMP threads
export MKL_NUM_THREADS=8            # MKL threads
```

### Runtime Configuration
```python
import torch

# GPU Memory Management
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()

# Performance Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

## Container Installation

### Docker
```bash
# Build and run
docker build -t yalgo-s .
docker run --gpus all yalgo-s

# Or use pre-built image
docker run --gpus all yalgo-s:latest
```

### Podman
```bash
podman build -t yalgo-s .
podman run --device nvidia.com/gpu=all yalgo-s
```

## Cloud Installation

### AWS EC2
```bash
# GPU instance setup
aws ec2 run-instances --instance-type p3.2xlarge --image-id ami-12345678

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos
pip install -e .
```

### Google Cloud
```bash
# GPU instance
gcloud compute instances create yalgo-s-instance \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos
pip install -e .
```

### Azure
```bash
# GPU VM
az vm create --name yalgo-s-vm --size Standard_NC6 --image Ubuntu2204

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/badpirogrammer2/yalgo-s.git
cd yalgo-s/ALGOs/New\ Algos
pip install -e .
```

## Support

For installation issues, please check:
- [GitHub Issues](https://github.com/badpirogrammer2/yalgo-s/issues)
- [Documentation](https://docs.yalgo-s.com)
- [Community Forum](https://github.com/badpirogrammer2/yalgo-s/discussions)
