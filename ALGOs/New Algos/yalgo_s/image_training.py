import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .agmohd.agmohd import AGMOHD  # Import the AGMOHD optimizer
import os

class ImageTrainer:
    def __init__(self, model=None, model_name=None, num_classes=10, device='auto', batch_size=32, max_epochs=10):
        if model is None and model_name is not None:
            self.model = self.load_pretrained_model(model_name, num_classes)
        elif model is not None:
            self.model = model
        else:
            raise ValueError("Either model or model_name must be provided")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.criterion = nn.CrossEntropyLoss()

    def load_pretrained_model(self, model_name, num_classes):
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)
        elif model_name == 'alexnet':
            model = torchvision.models.alexnet(pretrained=True)
            model.classifier[6] = nn.Linear(4096, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model

    def setup_data(self, dataset_name='CIFAR10', data_dir='./data', augmentation=True):
        if dataset_name == 'CIFAR10':
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4) if augmentation else transforms.ToTensor(),
                transforms.RandomHorizontalFlip() if augmentation else transforms.ToTensor(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
        elif dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        # Use AGMOHD for training
        trained_model = AGMOHD(self.model, self.train_loader, self.criterion, self.max_epochs)
        return trained_model

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

# Example usage
if __name__ == "__main__":
    # Example 1: Using a custom CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    print("Training with custom CNN on CIFAR-10:")
    model = SimpleCNN()
    trainer = ImageTrainer(model, batch_size=64, max_epochs=5)
    trainer.setup_data('CIFAR10', augmentation=True)
    trained_model = trainer.train()
    accuracy = trainer.evaluate()
    print(f"Test Accuracy: {accuracy}%")

    # Example 2: Using pre-trained ResNet18
    print("\nTraining with pre-trained ResNet18 on CIFAR-10:")
    trainer_resnet = ImageTrainer(model_name='resnet18', num_classes=10, batch_size=64, max_epochs=5)
    trainer_resnet.setup_data('CIFAR10', augmentation=True)
    trained_resnet = trainer_resnet.train()
    accuracy_resnet = trainer_resnet.evaluate()
    print(f"Test Accuracy: {accuracy_resnet}%")
