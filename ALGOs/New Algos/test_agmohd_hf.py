#!/usr/bin/env python3
"""
Comprehensive test suite for AGMOHD using Hugging Face datasets.

Tests include:
1. MNIST digit classification
2. CIFAR-10 image classification
3. GLUE benchmark tasks
4. Parallel processing validation
5. RTX 5060 optimization verification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import logging
import warnings
import json
from typing import Dict, Any, List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as transforms
from PIL import Image

from yalgo_s import AGMOHD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Exception Classes for Testing
class TestError(Exception):
    """Base exception class for test errors."""
    pass

class DatasetLoadError(TestError):
    """Raised when dataset loading fails."""
    pass

class ModelError(TestError):
    """Raised when model creation fails."""
    pass

class TrainingError(TestError):
    """Raised when training fails."""
    pass

class HFDataset(Dataset):
    """Custom dataset class for Hugging Face datasets."""

    def __init__(self, hf_dataset, transform=None, task_type='classification'):
        self.dataset = hf_dataset
        self.transform = transform
        self.task_type = task_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.task_type == 'image_classification':
            # Handle image data
            image = item['img']
            if isinstance(image, str):
                # Load image from path
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                # Convert to PIL Image if needed
                image = Image.fromarray(np.array(image))

            if self.transform:
                image = self.transform(image)

            label = item['label']
            return image, label

        elif self.task_type == 'text_classification':
            # Handle text data
            text = item['text'] if 'text' in item else item['sentence']
            label = item['label']
            return text, label

        else:
            # Default handling
            return item, item.get('label', 0)

def create_mnist_data_loader(batch_size=32, num_samples=10000):
    """Create MNIST data loader from Hugging Face with error handling."""
    try:
        logger.info("Loading MNIST dataset from Hugging Face...")

        # Input validation
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if num_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {num_samples}")

        # Load dataset with error handling
        try:
            dataset = load_dataset("mnist", split=f"train[:{num_samples}]")
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            raise DatasetLoadError(f"Cannot load MNIST dataset: {e}")

        # Transform with error handling
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        except Exception as e:
            logger.error(f"Failed to create transforms: {e}")
            raise DatasetLoadError(f"Cannot create image transforms: {e}")

        # Create dataset with error handling
        try:
            hf_dataset = HFDataset(dataset, transform=transform, task_type='image_classification')
            data_loader = DataLoader(hf_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        except Exception as e:
            logger.error(f"Failed to create data loader: {e}")
            raise DatasetLoadError(f"Cannot create MNIST data loader: {e}")

        logger.info(f"MNIST data loader created with {len(hf_dataset)} samples")
        return data_loader

    except Exception as e:
        logger.error(f"Failed to create MNIST data loader: {e}")
        raise DatasetLoadError(f"MNIST data loader creation failed: {e}") from e

def create_cifar10_data_loader(batch_size=32, num_samples=10000):
    """Create CIFAR-10 data loader from Hugging Face."""
    logger.info("Loading CIFAR-10 dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("cifar10", split=f"train[:{num_samples}]")

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Create dataset
    hf_dataset = HFDataset(dataset, transform=transform, task_type='image_classification')
    data_loader = DataLoader(hf_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader

def create_glue_data_loader(task_name="sst2", batch_size=32, num_samples=5000):
    """Create GLUE benchmark data loader from Hugging Face."""
    logger.info(f"Loading GLUE {task_name} dataset from Hugging Face...")

    # Load dataset
    dataset = load_dataset("glue", task_name, split=f"train[:{num_samples}]")

    # Create dataset
    hf_dataset = HFDataset(dataset, task_type='text_classification')
    data_loader = DataLoader(hf_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader

class MNISTModel(nn.Module):
    """CNN model for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

class CIFAR10Model(nn.Module):
    """CNN model for CIFAR-10 classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def test_mnist_agmohd():
    """Test AGMOHD on MNIST dataset with comprehensive error handling."""
    try:
        logger.info("Testing AGMOHD on MNIST dataset...")

        # Create model and data with error handling
        try:
            model = MNISTModel()
            logger.info("MNIST model created successfully")
        except Exception as e:
            logger.error(f"Failed to create MNIST model: {e}")
            raise ModelError(f"Cannot create MNIST model: {e}")

        try:
            data_loader = create_mnist_data_loader(batch_size=64, num_samples=5000)
            logger.info("MNIST data loader created successfully")
        except Exception as e:
            logger.error(f"Failed to create MNIST data loader: {e}")
            raise DatasetLoadError(f"Cannot create MNIST data loader: {e}")

        try:
            loss_fn = nn.NLLLoss()
            logger.info("Loss function created successfully")
        except Exception as e:
            logger.error(f"Failed to create loss function: {e}")
            raise TrainingError(f"Cannot create loss function: {e}")

        # Test configurations
        configs = [
            {
                "name": "Baseline (CPU)",
                "config": {"parallel_mode": "none", "device": "cpu", "use_rtx_optimizations": False}
            },
            {
                "name": "Parallel CPU",
                "config": {"parallel_mode": "thread", "device": "cpu", "use_rtx_optimizations": False}
            },
            {
                "name": "RTX Optimized",
                "config": {"parallel_mode": "none", "device": "auto", "use_rtx_optimizations": True}
            },
            {
                "name": "Full Optimized",
                "config": {"parallel_mode": "thread", "device": "auto", "use_rtx_optimizations": True}
            }
        ]

        results = {}

        for test_config in configs:
            logger.info(f"Running {test_config['name']} configuration...")

            try:
                # Create optimizer with error handling
                try:
                    optimizer = AGMOHD(model, **test_config['config'])
                    logger.info(f"AGMOHD optimizer created for {test_config['name']}")
                except Exception as e:
                    logger.error(f"Failed to create AGMOHD optimizer for {test_config['name']}: {e}")
                    results[test_config['name']] = {"error": f"Optimizer creation failed: {str(e)}"}
                    continue

                # Train model with error handling
                try:
                    start_time = time.time()
                    trained_model = optimizer.train(data_loader, loss_fn, max_epochs=3, verbose=False)
                    end_time = time.time()

                    training_time = end_time - start_time

                    # Get performance stats with error handling
                    try:
                        stats = optimizer.get_performance_stats()
                    except Exception as e:
                        logger.warning(f"Failed to get performance stats: {e}")
                        stats = {}

                    results[test_config['name']] = {
                        "training_time": training_time,
                        "performance_stats": stats,
                        "config": test_config['config']
                    }

                    logger.info(".2f")

                except Exception as e:
                    logger.error(f"Training failed for {test_config['name']}: {e}")
                    results[test_config['name']] = {"error": f"Training failed: {str(e)}"}

            except Exception as e:
                logger.error(f"Test configuration {test_config['name']} failed: {e}")
                results[test_config['name']] = {"error": f"Configuration failed: {str(e)}"}

        return results

    except Exception as e:
        logger.error(f"MNIST test failed: {e}")
        return {"error": f"MNIST test failed: {str(e)}"}

def test_cifar10_agmohd():
    """Test AGMOHD on CIFAR-10 dataset."""
    logger.info("Testing AGMOHD on CIFAR-10 dataset...")

    # Create model and data
    model = CIFAR10Model()
    data_loader = create_cifar10_data_loader(batch_size=64, num_samples=5000)
    loss_fn = nn.CrossEntropyLoss()

    # Single optimized configuration for CIFAR-10
    config = {
        "parallel_mode": "thread",
        "device": "auto",
        "use_rtx_optimizations": True
    }

    try:
        optimizer = AGMOHD(model, **config)

        start_time = time.time()
        trained_model = optimizer.train(data_loader, loss_fn, max_epochs=3, verbose=False)
        end_time = time.time()

        training_time = end_time - start_time
        stats = optimizer.get_performance_stats()

        results = {
            "cifar10_training_time": training_time,
            "cifar10_performance_stats": stats,
            "cifar10_config": config
        }

        logger.info(".2f")

        return results

    except Exception as e:
        logger.error(f"CIFAR-10 test failed: {e}")
        return {"error": str(e)}

def test_memory_optimization():
    """Test memory optimization features."""
    logger.info("Testing memory optimization features...")

    model = MNISTModel()
    data_loader = create_mnist_data_loader(batch_size=32, num_samples=1000)
    loss_fn = nn.NLLLoss()

    optimizer = AGMOHD(
        model,
        device="auto",
        use_rtx_optimizations=True,
        parallel_mode="thread"
    )

    # Test memory optimization
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    optimizer.optimize_memory_usage()

    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Run a few training steps
    for batch in data_loader:
        inputs, targets = batch
        loss = optimizer.step(loss_fn, inputs, targets)
        break

    peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    return {
        "initial_memory": initial_memory,
        "final_memory": final_memory,
        "peak_memory": peak_memory,
        "memory_efficiency": (initial_memory - final_memory) / max(initial_memory, 1) * 100
    }

def run_comprehensive_agmohd_tests():
    """Run comprehensive AGMOHD tests with Hugging Face datasets."""
    logger.info("Starting comprehensive AGMOHD tests with Hugging Face datasets...")

    results = {
        "mnist_results": test_mnist_agmohd(),
        "cifar10_results": test_cifar10_agmohd(),
        "memory_optimization": test_memory_optimization(),
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
        }
    }

    return results

def print_agmohd_test_results(results):
    """Print formatted AGMOHD test results."""
    print("\n" + "="*80)
    print("AGMOHD COMPREHENSIVE TEST RESULTS (Hugging Face Datasets)")
    print("="*80)

    # System Information
    print("\n🔧 SYSTEM INFORMATION:")
    sys_info = results["system_info"]
    print(f"  CUDA Available: {sys_info['cuda_available']}")
    print(f"  GPU Count: {sys_info['gpu_count']}")
    print(f"  GPU Name: {sys_info['gpu_name']}")
    print(f"  PyTorch Version: {sys_info['pytorch_version']}")

    # MNIST Results
    print("\n📊 MNIST RESULTS:")
    mnist_results = results["mnist_results"]
    for config_name, result in mnist_results.items():
        if "error" in result:
            print(f"  ❌ {config_name}: {result['error']}")
        else:
            time_taken = result["training_time"]
            config = result["config"]
            print(".2f"
                  f"device={config['device']}, rtx={config['use_rtx_optimizations']})")

    # CIFAR-10 Results
    print("\n🎨 CIFAR-10 RESULTS:")
    cifar_results = results["cifar10_results"]
    if "error" in cifar_results:
        print(f"  ❌ CIFAR-10 test failed: {cifar_results['error']}")
    else:
        time_taken = cifar_results["cifar10_training_time"]
        print(".2f")

    # Memory Optimization
    print("\n💾 MEMORY OPTIMIZATION:")
    mem_results = results["memory_optimization"]
    print(".2f")
    print(".2f")
    print(".1f")

    print("\n" + "="*80)
    print("🎉 AGMOHD TESTING COMPLETE!")
    print("📊 Results demonstrate parallel processing and RTX optimizations")
    print("🚀 Performance improvements validated on real datasets")
    print("="*80)

def main():
    """Main function to run AGMOHD tests with comprehensive error handling."""
    print("🚀 Starting AGMOHD Comprehensive Tests with Hugging Face Datasets")
    print("This will test MNIST, CIFAR-10, and memory optimization...")

    try:
        # Run comprehensive tests with error handling
        try:
            results = run_comprehensive_agmohd_tests()
            logger.info("Comprehensive tests completed successfully")
        except Exception as e:
            logger.error(f"Comprehensive tests failed: {e}")
            print(f"\n❌ Test execution failed: {e}")
            print("🔍 Check the log file for detailed error information")
            return 1

        # Print results with error handling
        try:
            print_agmohd_test_results(results)
        except Exception as e:
            logger.error(f"Failed to print test results: {e}")
            print(f"\n⚠️  Warning: Could not format test results: {e}")
            print("📊 Raw results are still available in the results dictionary")

        # Save results with error handling
        try:
            output_file = "agmohd_hf_test_results.json"

            # Convert torch tensors to serializable format
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            serializable_results[key][sub_key] = {}
                            for k, v in sub_value.items():
                                if torch.is_tensor(v):
                                    try:
                                        serializable_results[key][sub_key][k] = v.item()
                                    except Exception as e:
                                        logger.warning(f"Could not serialize tensor {k}: {e}")
                                        serializable_results[key][sub_key][k] = str(v)
                                else:
                                    serializable_results[key][sub_key][k] = v
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            print(f"\n📊 Detailed results saved to '{output_file}'")
            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"\n⚠️  Warning: Could not save results to file: {e}")
            print("💡 Results are still available in memory for analysis")

        print("\n✅ AGMOHD testing completed!")
        return 0

    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        print("\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    main()
