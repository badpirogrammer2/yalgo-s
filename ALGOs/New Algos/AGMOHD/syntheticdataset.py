#Input: 10-dimensional features.
#Output: A single continuous target value.
#Use case: Test the algorithm's ability to learn linear relationships.



import torch
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic regression data
def generate_regression_data(num_samples=100, input_dim=10, noise_std=0.1):
    # True weights and bias
    true_weights = torch.randn(input_dim, 1)
    true_bias = torch.randn(1)

    # Generate input features
    X = torch.randn(num_samples, input_dim)

    # Generate target values (with noise)
    y = X @ true_weights + true_bias + noise_std * torch.randn(num_samples, 1)

    return X, y

# Create dataset and data loader
X, y = generate_regression_data(num_samples=1000, input_dim=10, noise_std=0.1)
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example usage with AGMOHD
model = SimpleModel()
loss_fn = nn.MSELoss()
trained_model = AGMOHD(model, data_loader, loss_fn, max_epochs=20)