import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from agmohd import AGMOHD

# Define multiple model architectures
class LinearModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class ClassificationMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# Generate synthetic data
def generate_regression_data(num_samples=1000, input_dim=10):
    X = torch.randn(num_samples, input_dim)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
    return X, y

def generate_classification_data(num_samples=1000, input_dim=2):
    X = torch.randn(num_samples, input_dim)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)
    return X, y

# Test function
def test_agmohd_on_model(model, data_loader, loss_fn, max_epochs=20):
    trained_model = AGMOHD(model, data_loader, loss_fn, max_epochs)
    # Compute final loss
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            outputs = trained_model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            count += 1
    return total_loss / count

# Test on multiple models
models_to_test = [
    ("Linear Regression", LinearModel(10, 1), generate_regression_data(1000, 10), nn.MSELoss()),
    ("MLP Regression", MLPModel(10, 50, 1), generate_regression_data(1000, 10), nn.MSELoss()),
    ("MLP Classification", ClassificationMLP(2, 10, 1), generate_classification_data(1000, 2), nn.BCELoss()),
]

results = {}
for name, model, (X, y), loss_fn in models_to_test:
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    final_loss = test_agmohd_on_model(model, data_loader, loss_fn, max_epochs=10)
    results[name] = final_loss
    print(f"{name}: Final Loss = {final_loss:.4f}")

print("\nTest Results:")
for name, loss in results.items():
    print(f"{name}: {loss:.4f}")
