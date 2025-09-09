import torch
import torch.nn as nn
import torch.optim as optim
import math

def AGMOHD(model, data_loader, loss_fn, max_epochs):
    # Initialize parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01) # Use SGD as a base, we'll modify the learning rate
    m = [torch.zeros_like(param) for param in model.parameters()]  # Momentum for each parameter
    beta = 0.9  # Initial momentum factor
    eta_base = 0.01  # Base learning rate
    alpha = 0.1  # Gradient norm sensitivity
    T = 10  # Cycle length for learning rate schedule (adjust as needed)

    for epoch in range(max_epochs):
        for batch in data_loader:
            # Compute loss and gradients
            optimizer.zero_grad()  # Important: Clear gradients before each batch
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()  # Compute gradients

            grad = [param.grad for param in model.parameters()] # Extract gradients

            # Hindrance Detection Mechanism (Placeholder - Implement your logic)
            if detect_hindrance(grad, loss, model.parameters()):
                model, m, beta, eta_base = mitigate_hindrance(model, m, beta, eta_base)
                optimizer = optim.SGD(model.parameters(), lr=eta_base) # Re-initialize optimizer with new learning rate

            # Update momentum
            beta_t = adapt_beta(grad, beta) # Adapt beta if needed
            for i, param in enumerate(model.parameters()):
              m[i] = beta_t * m[i] + (1 - beta_t) * grad[i] # Update momentum for each parameter

            # Update learning rate
            eta_t = eta_base * (1 + math.cos(math.pi * (epoch % T) / T)) / 2
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad])) #Calculate gradient norm
            eta_t = eta_t / (1 + alpha * grad_norm**2)

            # Update parameters (using the optimizer)
            for i, param in enumerate(model.parameters()):
              param.data = param.data - eta_t * m[i] # Manually update parameters based on calculated learning rate and momentum

        print(f"Epoch: {epoch}, Loss: {loss.item()}, Learning Rate: {eta_t}") # Print loss and learning rate

    return model

# Implemented functions
def detect_hindrance(grad, loss, theta):
    # Detect based on gradient norm and loss spikes
    grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
    if grad_norm > 10 or loss.item() > 1e5:  # Adjusted thresholds
        return True
    return False

def mitigate_hindrance(model, m, beta, eta_base):
    # Reduce learning rate and reset momentum
    eta_base *= 0.5
    m = [torch.zeros_like(param) for param in model.parameters()]
    return model, m, beta, eta_base

def adapt_beta(grad, beta):
    # Adapt beta based on gradient variance
    grad_flat = torch.cat([g.view(-1) for g in grad])
    variance = torch.var(grad_flat)
    if variance > 1.0:
        beta = min(beta + 0.1, 0.99)
    else:
        beta = max(beta - 0.01, 0.8)
    return beta

# Example usage (replace with your model, data, and loss function):
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
data = [(torch.randn(1, 10), torch.randn(1, 1)) for _ in range(100)] # Example data
data_loader = torch.utils.data.DataLoader(data, batch_size=16)

loss_fn = nn.MSELoss()

trained_model = AGMOHD(model, data_loader, loss_fn, max_epochs=20)
