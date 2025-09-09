#!/usr/bin/env python3
"""
Test script for parallel processing and RTX 5060 optimizations in YALGO-S algorithms.

This script demonstrates:
1. AGMOHD with parallel processing and RTX optimizations
2. POIC-NET with multi-GPU support and batch processing
3. Performance comparisons between different configurations
4. Memory usage optimization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torchvision.transforms as transforms

# Import YALGO-S algorithms
from yalgo_s import AGMOHD, POICNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_data(num_samples=1000, input_dim=784, output_dim=10):
    """Create synthetic dataset for testing."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    return data_loader

def create_synthetic_images(num_images=10, size=(224, 224)):
    """Create synthetic images for POIC-NET testing."""
    images = []
    for _ in range(num_images):
        # Create random RGB image
        img_array = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images

def test_agmohd_parallel():
    """Test AGMOHD with different parallel configurations."""
    logger.info("Testing AGMOHD with parallel processing...")

    # Define model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    data_loader = create_synthetic_data()
    loss_fn = nn.CrossEntropyLoss()

    # Test configurations
    configs = [
        {"parallel_mode": "none", "device": "cpu", "use_rtx_optimizations": False},
        {"parallel_mode": "thread", "device": "cpu", "use_rtx_optimizations": False},
        {"parallel_mode": "none", "device": "auto", "use_rtx_optimizations": True},
        {"parallel_mode": "thread", "device": "auto", "use_rtx_optimizations": True},
    ]

    results = {}

    for i, config in enumerate(configs):
        logger.info(f"Testing configuration {i+1}: {config}")

        try:
            optimizer = AGMOHD(model, **config)

            start_time = time.time()
            trained_model = optimizer.train(data_loader, loss_fn, max_epochs=2, verbose=False)
            end_time = time.time()

            training_time = end_time - start_time
            stats = optimizer.get_performance_stats()

            results[f"config_{i+1}"] = {
                "config": config,
                "training_time": training_time,
                "performance_stats": stats
            }

            logger.info(".2f")

        except Exception as e:
            logger.error(f"Configuration {i+1} failed: {e}")
            results[f"config_{i+1}"] = {"error": str(e)}

    return results

def test_poic_net_parallel():
    """Test POIC-NET with parallel processing and multi-GPU."""
    logger.info("Testing POIC-NET with parallel processing...")

    images = create_synthetic_images(5)
    texts = [
        "A partially visible object in the scene",
        "Car with obscured license plate",
        "Person behind a pole",
        "Animal partially hidden by grass",
        "Object with missing parts"
    ]

    # Test configurations
    configs = [
        {"parallel_mode": "none", "device": "cpu", "use_rtx_optimizations": False},
        {"parallel_mode": "thread", "device": "auto", "use_rtx_optimizations": True},
        {"parallel_mode": "async", "device": "auto", "use_rtx_optimizations": True},
    ]

    results = {}

    for i, config in enumerate(configs):
        logger.info(f"Testing POIC-NET configuration {i+1}: {config}")

        try:
            poic_net = POICNet(**config)

            # Test single image processing
            start_time = time.time()
            objects, scores = poic_net(images[0])
            single_time = time.time() - start_time

            # Test batch processing
            start_time = time.time()
            batch_results = []
            for img, text in zip(images, texts):
                obj, score = poic_net((img, text), modality="multimodal")
                batch_results.append((obj, score))
            batch_time = time.time() - start_time

            # Get performance stats
            stats = poic_net.get_performance_stats()

            results[f"poic_config_{i+1}"] = {
                "config": config,
                "single_image_time": single_time,
                "batch_time": batch_time,
                "num_objects_detected": len([r for r in batch_results if r[0]]),
                "performance_stats": stats
            }

            logger.info(".3f")

        except Exception as e:
            logger.error(f"POIC-NET configuration {i+1} failed: {e}")
            results[f"poic_config_{i+1}"] = {"error": str(e)}

    return results

def test_multi_gpu_support():
    """Test multi-GPU support if available."""
    logger.info("Testing multi-GPU support...")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping multi-GPU test")
        return {"multi_gpu": "CUDA not available"}

    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s)")

    if gpu_count < 2:
        logger.info("Single GPU system, testing single GPU optimizations")
        poic_net = POICNet(device="cuda:0", use_rtx_optimizations=True)
        poic_net.optimize_memory_usage()

        # Test memory optimization
        stats = poic_net.get_performance_stats()
        return {"single_gpu_optimized": stats}
    else:
        logger.info("Multi-GPU system detected, testing parallel processing")

        # Test with multiple GPUs
        gpu_ids = list(range(min(2, gpu_count)))
        poic_net = POICNet(device="cuda:0", use_rtx_optimizations=True)
        poic_net.enable_multi_gpu(gpu_ids)

        # Test with multi-GPU setup
        images = create_synthetic_images(10)

        start_time = time.time()
        results = []
        for img in images:
            obj, score = poic_net(img)
            results.append((obj, score))
        multi_gpu_time = time.time() - start_time

        stats = poic_net.get_performance_stats()

        return {
            "multi_gpu_time": multi_gpu_time,
            "gpus_used": gpu_ids,
            "performance_stats": stats
        }

def benchmark_algorithms():
    """Comprehensive benchmark of all algorithms."""
    logger.info("Running comprehensive benchmark...")

    results = {
        "agmohd_results": test_agmohd_parallel(),
        "poic_net_results": test_poic_net_parallel(),
        "multi_gpu_results": test_multi_gpu_support(),
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "cpu_count": torch.get_num_threads(),
        }
    }

    return results

def print_benchmark_results(results):
    """Print formatted benchmark results."""
    print("\n" + "="*80)
    print("YALGO-S PARALLEL PROCESSING & RTX 5060 OPTIMIZATION BENCHMARK RESULTS")
    print("="*80)

    # System Information
    print("\n🔧 SYSTEM INFORMATION:")
    sys_info = results["system_info"]
    print(f"  CUDA Available: {sys_info['cuda_available']}")
    print(f"  GPU Count: {sys_info['gpu_count']}")
    print(f"  GPU Name: {sys_info['gpu_name']}")
    print(f"  CPU Threads: {sys_info['cpu_count']}")

    # AGMOHD Results
    print("\n🎯 AGMOHD OPTIMIZER RESULTS:")
    agmohd_results = results["agmohd_results"]
    for config_name, result in agmohd_results.items():
        if "error" in result:
            print(f"  ❌ {config_name}: {result['error']}")
        else:
            config = result["config"]
            time_taken = result["training_time"]
            print(".2f"
                  f"device={config['device']}, rtx={config['use_rtx_optimizations']})")

    # POIC-NET Results
    print("\n🔍 POIC-NET RESULTS:")
    poic_results = results["poic_net_results"]
    for config_name, result in poic_results.items():
        if "error" in result:
            print(f"  ❌ {config_name}: {result['error']}")
        else:
            config = result["config"]
            single_time = result["single_image_time"]
            batch_time = result["batch_time"]
            objects_detected = result["num_objects_detected"]
            print(".3f"
                  f"detected={objects_detected}, "
                  f"device={config['device']}, rtx={config['use_rtx_optimizations']})")

    # Multi-GPU Results
    print("\n🚀 MULTI-GPU RESULTS:")
    multi_gpu = results["multi_gpu_results"]
    if "error" in multi_gpu:
        print(f"  ❌ Multi-GPU test failed: {multi_gpu['error']}")
    elif "single_gpu_optimized" in multi_gpu:
        print("  ✅ Single GPU with RTX optimizations enabled")
        stats = multi_gpu["single_gpu_optimized"]
        if "current_memory" in stats:
            print(".2f")
    elif "multi_gpu_time" in multi_gpu:
        gpu_time = multi_gpu["multi_gpu_time"]
        gpus_used = multi_gpu["gpus_used"]
        print(".3f"
              f"GPUs={gpus_used})")

    print("\n" + "="*80)
    print("🎉 BENCHMARK COMPLETE!")
    print("💡 Use RTX optimizations for best performance on RTX 5060 and newer GPUs")
    print("🔥 Enable parallel processing for improved throughput")
    print("🚀 Consider multi-GPU setup for large-scale processing")
    print("="*80)

def main():
    """Main function to run all tests."""
    print("🚀 Starting YALGO-S Parallel Processing & RTX 5060 Optimization Tests")
    print("This may take several minutes depending on your hardware...")

    try:
        # Run comprehensive benchmark
        results = benchmark_algorithms()

        # Print results
        print_benchmark_results(results)

        # Save results to file
        import json
        with open("benchmark_results.json", "w") as f:
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
                                    serializable_results[key][sub_key][k] = v.item()
                                else:
                                    serializable_results[key][sub_key][k] = v
                        else:
                            serializable_results[key][sub_key] = sub_value
                else:
                    serializable_results[key] = value

            json.dump(serializable_results, f, indent=2)

        print("\n📊 Results saved to 'benchmark_results.json'")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
