#!/usr/bin/env python3
"""
YALGO-S Image Training Example

This example demonstrates how to use the ImageTrainer class for training
image classification models with the AGMOHD optimizer.

Features demonstrated:
- Custom CNN model training
- Pre-trained model fine-tuning (ResNet18)
- Data augmentation
- GPU acceleration
- Performance evaluation
"""

import torch
import torch.nn as nn
from yalgo_s import ImageTrainer

def main():
    print("🚀 YALGO-S Image Training Example")
    print("=" * 50)

    # Example 1: Training a custom CNN on CIFAR-10
    print("\n📚 Example 1: Custom CNN on CIFAR-10")
    print("-" * 40)

    # Define a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = self.dropout(x)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Create trainer with custom model
    custom_trainer = ImageTrainer(
        model=SimpleCNN(),
        batch_size=128,
        max_epochs=10
    )

    # Setup data with augmentation
    custom_trainer.setup_data(
        dataset_name='CIFAR10',
        data_dir='./data',
        augmentation=True
    )

    print("Training custom CNN...")
    trained_custom = custom_trainer.train()
    custom_accuracy = custom_trainer.evaluate()
    print(f"Custom CNN Test Accuracy: {custom_accuracy:.2f}%")

    # Example 2: Fine-tuning pre-trained ResNet18
    print("\n🤖 Example 2: Fine-tuning ResNet18 on CIFAR-10")
    print("-" * 50)

    # Create trainer with pre-trained model
    resnet_trainer = ImageTrainer(
        model_name='resnet18',
        num_classes=10,
        batch_size=64,
        max_epochs=5
    )

    # Setup data
    resnet_trainer.setup_data(
        dataset_name='CIFAR10',
        data_dir='./data',
        augmentation=True
    )

    print("Fine-tuning ResNet18...")
    trained_resnet = resnet_trainer.train()
    resnet_accuracy = resnet_trainer.evaluate()
    print(f"ResNet18 Test Accuracy: {resnet_accuracy:.2f}%")

    # Example 3: Training on MNIST with VGG16
    print("\n📊 Example 3: VGG16 on MNIST")
    print("-" * 30)

    # Note: MNIST is grayscale, so we'll convert to 3 channels
    import torchvision.transforms as transforms

    # Custom transform to convert grayscale to RGB
    class GrayscaleToRgb:
        def __call__(self, img):
            return img.convert('RGB')

    # Create trainer with VGG16
    vgg_trainer = ImageTrainer(
        model_name='vgg16',
        num_classes=10,
        batch_size=64,
        max_epochs=3
    )

    # Setup MNIST data with custom transforms
    mnist_transform = transforms.Compose([
        GrayscaleToRgb(),
        transforms.Resize((224, 224)),  # VGG16 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Manually setup data for MNIST with custom transform
    import torchvision
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=mnist_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=mnist_transform
    )

    from torch.utils.data import DataLoader
    vgg_trainer.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    vgg_trainer.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Training VGG16 on MNIST...")
    trained_vgg = vgg_trainer.train()
    vgg_accuracy = vgg_trainer.evaluate()
    print(f"VGG16 Test Accuracy: {vgg_accuracy:.2f}%")

    # Summary
    print("\n📈 Training Summary")
    print("=" * 20)
    print(f"{'Model':<20} {'Accuracy':<10}")
    print("-" * 35)
    print(f"{'Custom CNN':<20} {custom_accuracy:.2f}%")
    print(f"{'ResNet18':<20} {resnet_accuracy:.2f}%")
    print(f"{'VGG16':<20} {vgg_accuracy:.2f}%")

    print("\n✅ Image training examples completed!")
    print("💡 Tips:")
    print("   - Use data augmentation for better generalization")
    print("   - Pre-trained models often perform better with fine-tuning")
    print("   - Adjust batch size and epochs based on your hardware")
    print("   - AGMOHD optimizer provides adaptive learning for stable training")

if __name__ == "__main__":
    main()
