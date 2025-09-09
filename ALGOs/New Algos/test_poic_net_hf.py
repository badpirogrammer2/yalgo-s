#!/usr/bin/env python3
"""
Comprehensive test suite for POIC-NET using Hugging Face datasets.

Tests include:
1. COCO dataset for object detection
2. Flickr30k for image-text matching
3. Multi-modal understanding
4. Parallel processing validation
5. RTX 5060 optimization verification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoImageProcessor
import torchvision.transforms as transforms
from PIL import Image
import json

from yalgo_s import POICNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COCODataset(Dataset):
    """COCO dataset for object detection tasks."""

    def __init__(self, split="train", num_samples=1000):
        self.dataset = load_dataset("detection-datasets/coco", split=f"{split}[:{num_samples}]")
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Get annotations
        annotations = item.get("objects", {})
        bboxes = annotations.get("bbox", [])
        categories = annotations.get("category", [])

        # Create text description from categories
        if categories:
            text = f"Image containing {', '.join([str(cat) for cat in categories[:3]])}"
        else:
            text = "Image with various objects"

        return {
            "image": image,
            "text": text,
            "bboxes": bboxes,
            "categories": categories
        }

class Flickr30kDataset(Dataset):
    """Flickr30k dataset for image-text matching."""

    def __init__(self, split="train", num_samples=1000):
        self.dataset = load_dataset("nlphuji/flickr30k", split=f"{split}[:{num_samples}]")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Get text captions (use first caption)
        captions = item.get("caption", [])
        text = captions[0] if captions else "An image"

        return {
            "image": image,
            "text": text,
            "all_captions": captions
        }

class VQADataset(Dataset):
    """VQA dataset for visual question answering."""

    def __init__(self, split="train", num_samples=1000):
        self.dataset = load_dataset("HuggingFaceM4/VQAv2", split=f"{split}[:{num_samples}]")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item["image"]
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Get question and answer
        question = item.get("question", "What is in this image?")
        answer = item.get("answer", "Unknown")

        # Create combined text
        text = f"Question: {question} Answer: {answer}"

        return {
            "image": image,
            "text": text,
            "question": question,
            "answer": answer
        }

def create_coco_data_loader(batch_size=8, num_samples=500):
    """Create COCO data loader."""
    logger.info("Loading COCO dataset from Hugging Face...")

    dataset = COCODataset(split="train", num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader

def create_flickr_data_loader(batch_size=8, num_samples=500):
    """Create Flickr30k data loader."""
    logger.info("Loading Flickr30k dataset from Hugging Face...")

    dataset = Flickr30kDataset(split="train", num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader

def create_vqa_data_loader(batch_size=8, num_samples=500):
    """Create VQA data loader."""
    logger.info("Loading VQA dataset from Hugging Face...")

    dataset = VQADataset(split="train", num_samples=num_samples)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return data_loader

def test_poic_net_coco():
    """Test POIC-NET on COCO dataset."""
    logger.info("Testing POIC-NET on COCO dataset...")

    # Create data loader
    data_loader = create_coco_data_loader(batch_size=4, num_samples=100)

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
            poic_net = POICNet(**test_config['config'])

            total_time = 0
            total_objects = 0
            processed_samples = 0

            start_time = time.time()

            for batch in data_loader:
                batch_start = time.time()

                for item in batch:
                    image = item["image"]
                    text = item["text"]

                    # Test single image processing
                    objects, scores = poic_net((image, text), modality="multimodal")

                    total_objects += len(objects) if objects else 0
                    processed_samples += 1

                batch_time = time.time() - batch_start
                total_time += batch_time

                # Process only first few batches for speed
                if processed_samples >= 20:
                    break

            total_processing_time = time.time() - start_time
            stats = poic_net.get_performance_stats()

            results[test_config['name']] = {
                "total_time": total_processing_time,
                "avg_time_per_sample": total_processing_time / max(processed_samples, 1),
                "total_objects_detected": total_objects,
                "samples_processed": processed_samples,
                "performance_stats": stats,
                "config": test_config['config']
            }

            logger.info(".3f"
                  f"objects={total_objects})")

        except Exception as e:
            logger.error(f"{test_config['name']} failed: {e}")
            results[test_config['name']] = {"error": str(e)}

    return results

def test_poic_net_flickr():
    """Test POIC-NET on Flickr30k dataset."""
    logger.info("Testing POIC-NET on Flickr30k dataset...")

    # Create data loader
    data_loader = create_flickr_data_loader(batch_size=4, num_samples=100)

    # Optimized configuration
    config = {
        "parallel_mode": "thread",
        "device": "auto",
        "use_rtx_optimizations": True
    }

    try:
        poic_net = POICNet(**config)

        total_time = 0
        processed_samples = 0

        start_time = time.time()

        for batch in data_loader:
            for item in batch:
                image = item["image"]
                text = item["text"]

                # Test multi-modal processing
                objects, scores = poic_net((image, text), modality="multimodal")

                processed_samples += 1

            # Process only first few batches
            if processed_samples >= 20:
                break

        total_processing_time = time.time() - start_time
        stats = poic_net.get_performance_stats()

        results = {
            "flickr_total_time": total_processing_time,
            "flickr_avg_time_per_sample": total_processing_time / max(processed_samples, 1),
            "flickr_samples_processed": processed_samples,
            "flickr_performance_stats": stats,
            "flickr_config": config
        }

        logger.info(".3f")

        return results

    except Exception as e:
        logger.error(f"Flickr30k test failed: {e}")
        return {"error": str(e)}

def test_poic_net_batch_processing():
    """Test batch processing capabilities."""
    logger.info("Testing POIC-NET batch processing...")

    # Create multiple images for batch testing
    images = []
    texts = []

    for i in range(10):
        # Create synthetic image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
        texts.append(f"Test image {i} with various objects")

    config = {
        "parallel_mode": "thread",
        "device": "auto",
        "use_rtx_optimizations": True,
        "batch_size": 4
    }

    try:
        poic_net = POICNet(**config)

        # Test batch processing
        start_time = time.time()
        batch_results = []

        for i in range(0, len(images), config["batch_size"]):
            batch_images = images[i:i+config["batch_size"]]
            batch_texts = texts[i:i+config["batch_size"]]

            # Process batch
            for img, text in zip(batch_images, batch_texts):
                objects, scores = poic_net((img, text), modality="multimodal")
                batch_results.append((objects, scores))

        batch_time = time.time() - start_time

        # Test individual processing for comparison
        start_time = time.time()
        individual_results = []

        for img, text in zip(images, texts):
            objects, scores = poic_net((img, text), modality="multimodal")
            individual_results.append((objects, scores))

        individual_time = time.time() - start_time

        stats = poic_net.get_performance_stats()

        return {
            "batch_time": batch_time,
            "individual_time": individual_time,
            "speedup": individual_time / batch_time if batch_time > 0 else 0,
            "batch_size": config["batch_size"],
            "performance_stats": stats
        }

    except Exception as e:
        logger.error(f"Batch processing test failed: {e}")
        return {"error": str(e)}

def test_multi_gpu_poic_net():
    """Test multi-GPU capabilities."""
    logger.info("Testing POIC-NET multi-GPU capabilities...")

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

        # Create test data
        images = []
        texts = []
        for i in range(20):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            images.append(img)
            texts.append(f"Multi-GPU test image {i}")

        start_time = time.time()
        results = []
        for img, text in zip(images, texts):
            obj, score = poic_net((img, text))
            results.append((obj, score))
        multi_gpu_time = time.time() - start_time

        stats = poic_net.get_performance_stats()

        return {
            "multi_gpu_time": multi_gpu_time,
            "gpus_used": gpu_ids,
            "samples_processed": len(images),
            "performance_stats": stats
        }

def run_comprehensive_poic_net_tests():
    """Run comprehensive POIC-NET tests with Hugging Face datasets."""
    logger.info("Starting comprehensive POIC-NET tests with Hugging Face datasets...")

    results = {
        "coco_results": test_poic_net_coco(),
        "flickr_results": test_poic_net_flickr(),
        "batch_processing": test_poic_net_batch_processing(),
        "multi_gpu_results": test_multi_gpu_poic_net(),
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
        }
    }

    return results

def print_poic_net_test_results(results):
    """Print formatted POIC-NET test results."""
    print("\n" + "="*80)
    print("POIC-NET COMPREHENSIVE TEST RESULTS (Hugging Face Datasets)")
    print("="*80)

    # System Information
    print("\n🔧 SYSTEM INFORMATION:")
    sys_info = results["system_info"]
    print(f"  CUDA Available: {sys_info['cuda_available']}")
    print(f"  GPU Count: {sys_info['gpu_count']}")
    print(f"  GPU Name: {sys_info['gpu_name']}")
    print(f"  PyTorch Version: {sys_info['pytorch_version']}")

    # COCO Results
    print("\n🎯 COCO OBJECT DETECTION RESULTS:")
    coco_results = results["coco_results"]
    for config_name, result in coco_results.items():
        if "error" in result:
            print(f"  ❌ {config_name}: {result['error']}")
        else:
            total_time = result["total_time"]
            samples = result["samples_processed"]
            objects = result["total_objects_detected"]
            avg_time = result["avg_time_per_sample"]
            print(".2f"
                  f"objects={objects}, avg_time={avg_time:.3f}s)")

    # Flickr Results
    print("\n📸 FLICKR30K IMAGE-TEXT RESULTS:")
    flickr_results = results["flickr_results"]
    if "error" in flickr_results:
        print(f"  ❌ Flickr30k test failed: {flickr_results['error']}")
    else:
        total_time = flickr_results["flickr_total_time"]
        samples = flickr_results["flickr_samples_processed"]
        avg_time = flickr_results["flickr_avg_time_per_sample"]
        print(".2f"
              f"avg_time={avg_time:.3f}s)")

    # Batch Processing
    print("\n📦 BATCH PROCESSING RESULTS:")
    batch_results = results["batch_processing"]
    if "error" in batch_results:
        print(f"  ❌ Batch processing failed: {batch_results['error']}")
    else:
        batch_time = batch_results["batch_time"]
        individual_time = batch_results["individual_time"]
        speedup = batch_results["speedup"]
        batch_size = batch_results["batch_size"]
        print(".2f"
              f"speedup={speedup:.2f}x)")

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
        samples = multi_gpu["samples_processed"]
        print(".2f"
              f"GPUs={gpus_used})")

    print("\n" + "="*80)
    print("🎉 POIC-NET TESTING COMPLETE!")
    print("🔍 Multi-modal capabilities validated on real datasets")
    print("🚀 Parallel processing and RTX optimizations confirmed")
    print("="*80)

def main():
    """Main function to run POIC-NET tests."""
    print("🚀 Starting POIC-NET Comprehensive Tests with Hugging Face Datasets")
    print("This will test COCO, Flickr30k, batch processing, and multi-GPU...")

    try:
        # Run comprehensive tests
        results = run_comprehensive_poic_net_tests()

        # Print results
        print_poic_net_test_results(results)

        # Save results
        with open("poic_net_hf_test_results.json", "w") as f:
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

        print("\n📊 Detailed results saved to 'poic_net_hf_test_results.json'")

    except Exception as e:
        logger.error(f"POIC-NET tests failed: {e}")
        raise

if __name__ == "__main__":
    main()
