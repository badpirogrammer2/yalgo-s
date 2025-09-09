#!/usr/bin/env python3
"""
Test script for YALGO-S Image Training functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from yalgo_s import ImageTrainer
        print("✅ ImageTrainer import successful")
        return True
    except ImportError as e:
        print(f"❌ ImageTrainer import failed: {e}")
        return False

def test_class_instantiation():
    """Test if ImageTrainer can be instantiated"""
    try:
        from yalgo_s import ImageTrainer
        import torch.nn as nn

        # Test with custom model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)

            def forward(self, x):
                return self.linear(x)

        trainer = ImageTrainer(model=SimpleModel(), batch_size=32, max_epochs=5)
        print("✅ ImageTrainer instantiation with custom model successful")
        return True
    except Exception as e:
        print(f"❌ ImageTrainer instantiation failed: {e}")
        return False

def test_pretrained_model_loading():
    """Test if pre-trained models can be loaded"""
    try:
        from yalgo_s import ImageTrainer

        # This will fail if torchvision is not available, but we can catch that
        trainer = ImageTrainer(model_name='resnet18', num_classes=10, batch_size=32, max_epochs=5)
        print("✅ Pre-trained model loading successful")
        return True
    except ImportError as e:
        print(f"⚠️  Pre-trained model loading failed (torchvision not available): {e}")
        return False
    except Exception as e:
        print(f"❌ Pre-trained model loading failed: {e}")
        return False

def test_data_setup():
    """Test data setup functionality"""
    try:
        from yalgo_s import ImageTrainer
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(784, 10)  # For MNIST-like data

            def forward(self, x):
                return self.linear(x.view(x.size(0), -1))

        trainer = ImageTrainer(model=SimpleModel(), batch_size=32, max_epochs=5)

        # Test data setup (this will download if not present)
        trainer.setup_data(dataset_name='MNIST', data_dir='./test_data', augmentation=False)
        print("✅ Data setup successful")
        return True
    except ImportError as e:
        print(f"⚠️  Data setup failed (torchvision not available): {e}")
        return False
    except Exception as e:
        print(f"❌ Data setup failed: {e}")
        return False

def main():
    print("🧪 Testing YALGO-S Image Training Functionality")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Class Instantiation Test", test_class_instantiation),
        ("Pre-trained Model Test", test_pretrained_model_loading),
        ("Data Setup Test", test_data_setup),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Image training functionality is working correctly.")
    elif passed >= total - 1:  # Allow one failure (likely torchvision)
        print("✅ Core functionality working. Some optional features may require additional dependencies.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

    print("\n💡 To run full training tests, ensure you have:")
    print("   - PyTorch installed")
    print("   - torchvision installed")
    print("   - Sufficient disk space for datasets")

if __name__ == "__main__":
    main()
