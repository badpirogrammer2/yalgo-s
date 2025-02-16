#Input: 2-dimensional features (for visualization).
#Output: Binary labels (0 or 1).
#Use case: Test the algorithm's ability to learn non-linear decision boundaries

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate synthetic classification data
def generate_classification_data(num_samples=1000, noise=0.2):
    X, y = make_moons(n_samples=num_samples, noise=noise, random_state=42)
    X = StandardScaler().fit_transform(X)  # Normalize data
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y

# Create dataset and data loader
X, y = generate_classification_data(num_samples=1000, noise=0.2)
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Example usage with AGMOHD
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

model = ClassificationModel()
loss_fn = nn.BCELoss()  # Binary cross-entropy loss
trained_model = AGMOHD(model, data_loader, loss_fn, max_epochs=20)