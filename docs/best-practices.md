# YALGO-S Best Practices

## Overview

This guide covers best practices for using YALGO-S algorithms effectively in your machine learning projects.

## 🧠 AGMOHD Optimizer

### Learning Rate Selection
```python
# Good starting points
optimizer = AGMOHD(model, lr=0.01)  # General purpose
optimizer = AGMOHD(model, lr=0.001)  # Fine-tuning
optimizer = AGMOHD(model, lr=0.1)   # Large datasets
```

### Beta Parameter Tuning
```python
# Conservative momentum (stable training)
optimizer = AGMOHD(model, lr=0.01, beta=0.95)

# Aggressive momentum (faster convergence)
optimizer = AGMOHD(model, lr=0.01, beta=0.85)
```

### RTX Optimization
```python
# Enable RTX optimizations for RTX 40-series GPUs
optimizer = AGMOHD(
    model,
    device='cuda',
    use_rtx_optimizations=True
)
```

## 🖼️ Image Training

### Data Preparation
```python
# Always use data augmentation for better generalization
trainer = ImageTrainer(model_name='resnet18')
trainer.setup_data('CIFAR10', augmentation=True)

# Use appropriate batch sizes
trainer = ImageTrainer(batch_size=64)  # RTX 3060+
trainer = ImageTrainer(batch_size=32)  # RTX 2060
trainer = ImageTrainer(batch_size=16)  # Limited VRAM
```

### Model Selection
```python
# Start with pre-trained models
trainer = ImageTrainer(model_name='resnet18')    # Good balance
trainer = ImageTrainer(model_name='efficientnet_b0')  # Lightweight
trainer = ImageTrainer(model_name='vgg16')       # Detailed features

# Use custom models for specific requirements
class CustomModel(nn.Module):
    # Implement your architecture
    pass
```

### Training Optimization
```python
# Enable mixed precision for faster training
trainer = ImageTrainer(mixed_precision=True)

# Use gradient clipping to prevent exploding gradients
trainer = ImageTrainer(grad_clip=1.0)

# Implement early stopping
trainer.train(early_stopping=True, patience=10)
```

## 🔍 POIC-NET

### Model Configuration
```python
# Choose appropriate backbone
poic_net = POICNet(
    image_model_name="resnet50",    # Balanced performance
    text_model_name="bert"          # General purpose
)

# Use efficient models for real-time applications
poic_net = POICNet(
    image_model_name="mobilenet_v2",
    text_model_name="distilbert"
)
```

### Threshold Tuning
```python
# Adjust detection threshold based on use case
poic_net = POICNet(threshold=0.8)  # High precision
poic_net = POICNet(threshold=0.5)  # High recall
poic_net = POICNet(threshold=0.3)  # Maximum detections
```

### Multi-Modal Processing
```python
# Always provide both image and text when available
objects, scores = poic_net((image, text_description))

# Use image-only for faster processing
objects, scores = poic_net(image, modality="image")
```

## 🧠 ARCE

### Network Configuration
```python
# Start with reasonable defaults
arce = ARCE(
    input_dim=100,
    vigilance_base=0.8,
    learning_rate=0.1
)

# Adjust for your data characteristics
arce = ARCE(
    input_dim=data_dim,
    vigilance_base=0.7,  # Lower for noisy data
    max_categories=50    # Limit categories for memory
)
```

### Context Engineering
```python
# Provide rich contextual information
context = {
    'temporal': datetime.now().hour,
    'spatial': get_location(),
    'environmental': get_weather(),
    'user_state': get_user_activity()
}

# Use consistent context keys
category = arce.learn(data, context)
```

### Vigilance Adaptation
```python
# Adapt vigilance based on context stability
if context_is_stable(context):
    arce.vigilance_base = 0.9  # Higher vigilance
else:
    arce.vigilance_base = 0.6  # Lower vigilance
```

## 🚀 Performance Optimization

### GPU Memory Management
```python
import torch

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)

# Empty cache regularly
torch.cuda.empty_cache()

# Use gradient accumulation for large batches
# Implement in custom training loop
```

### Multi-GPU Training
```python
# DataParallel for simple multi-GPU
model = nn.DataParallel(model)

# DistributedDataParallel for advanced setups
# Use torch.distributed for large-scale training
```

### CPU Optimization
```python
# Set thread counts
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

# Use efficient data loading
from torch.utils.data import DataLoader
loader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

## 📊 Monitoring and Debugging

### Performance Monitoring
```python
# Monitor GPU usage
stats = optimizer.get_performance_stats()
print(f"GPU Memory: {stats['current_memory']:.2f} GB")
print(f"GPU Utilization: {stats['current_gpu_util']:.1f}%")

# Track training metrics
trainer = ImageTrainer(track_metrics=True)
history = trainer.train()
```

### Logging Best Practices
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log important events
logger = logging.getLogger(__name__)
logger.info("Training started")
logger.info(f"Epoch {epoch}: Loss = {loss:.4f}")
```

### Error Handling
```python
try:
    trained_model = trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size
        trainer.batch_size = trainer.batch_size // 2
        trained_model = trainer.train()
    else:
        raise
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

**CUDA Out of Memory**
```python
# Solutions in order of preference:
# 1. Reduce batch size
trainer = ImageTrainer(batch_size=16)

# 2. Use gradient accumulation
# 3. Use mixed precision training
# 4. Use smaller model
trainer = ImageTrainer(model_name='efficientnet_b0')

# 5. Reduce input size
# 6. Use CPU training (last resort)
```

**Slow Training**
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Use DataParallel
model = nn.DataParallel(model)

# Optimize data loading
loader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

**Poor Model Performance**
```python
# Check data quality
trainer.setup_data('CIFAR10', augmentation=True)

# Adjust learning rate
optimizer = AGMOHD(model, lr=0.001)

# Use better architecture
trainer = ImageTrainer(model_name='resnet50')

# Implement regularization
# Add dropout, batch normalization
```

**Memory Leaks**
```python
# Clear cache regularly
torch.cuda.empty_cache()

# Delete unused variables
del intermediate_results

# Use context managers
with torch.no_grad():
    # Inference code
```

## 📈 Scaling Best Practices

### Small Scale (Single GPU)
```python
# Use RTX optimizations
trainer = ImageTrainer(use_rtx_optimizations=True)

# Optimize batch size for your GPU
trainer = ImageTrainer(batch_size=64)  # RTX 3060
```

### Medium Scale (Multi-GPU)
```python
# Enable DataParallel
model = nn.DataParallel(model)

# Use larger batch sizes
trainer = ImageTrainer(batch_size=256)

# Optimize data loading
loader = DataLoader(dataset, num_workers=8, pin_memory=True)
```

### Large Scale (Distributed)
```python
# Use DistributedDataParallel
import torch.distributed as dist
# Initialize process group
# Wrap model with DDP

# Use gradient accumulation
# Implement custom training loop
```

## 🔒 Production Deployment

### Model Serialization
```python
# Save trained model
torch.save(model.state_dict(), 'model.pth')

# Load for inference
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### Inference Optimization
```python
# Use TorchScript for faster inference
scripted_model = torch.jit.script(model)

# Use ONNX for cross-platform deployment
torch.onnx.export(model, input_sample, 'model.onnx')
```

### Monitoring in Production
```python
# Log predictions
logger.info(f"Prediction: {prediction}, Confidence: {confidence}")

# Monitor latency
start_time = time.time()
result = model(input_data)
latency = time.time() - start_time
logger.info(f"Inference latency: {latency:.3f}s")
```

## 📚 Additional Resources

### Documentation
- [API Reference](ALGOs/New%20Algos/Readme.html)
- [Examples](examples/)
- [Troubleshooting Guide](docs/troubleshooting.md)

### Community
- [GitHub Issues](https://github.com/badpirogrammer2/yalgo-s/issues)
- [Discussions](https://github.com/badpirogrammer2/yalgo-s/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/yalgo-s)

### Performance Tuning
- [GPU Optimization Guide](ALGOs/New%20Algos/AGMOHD/yalgo_s_cross_platform.html)
- [Memory Management](docs/memory-optimization.md)
- [Benchmark Results](ALGOs/New%20Algos/AGMOHD/yalgo_s_performance_heatmap.html)

Remember: These are guidelines, not strict rules. Always profile your specific use case and adjust parameters accordingly for optimal performance.
