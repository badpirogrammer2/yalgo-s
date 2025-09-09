import torch
import torch.nn as nn
import torch.optim as optim
import math

class AGMOHD:
    """
    Adaptive Gradient Momentum with Hindrance Detection Optimizer

    Dynamically adjusts learning rates and momentum based on gradient analysis
    and training hindrance detection for improved convergence.

    Args:
        model (nn.Module): The neural network model to optimize
        lr (float): Initial learning rate (default: 0.01)
        beta (float): Initial momentum factor (default: 0.9)
        alpha (float): Gradient norm sensitivity (default: 0.1)
        T (int): Cycle length for learning rate schedule (default: 10)
    """

    def __init__(self, model, lr=0.01, beta=0.9, alpha=0.1, T=10):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.alpha = alpha
        self.T = T
        self.epoch = 0

        # Initialize momentum for each parameter
        self.m = [torch.zeros_like(param) for param in model.parameters()]

        # Optimizer for parameter updates
        self.optimizer = optim.SGD(model.parameters(), lr=lr)

    def detect_hindrance(self, grad, loss):
        """
        Detect training hindrances based on gradient norm and loss spikes.

        Args:
            grad: List of gradients for each parameter
            loss: Current loss value

        Returns:
            bool: True if hindrance detected
        """
        grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
        if grad_norm > 10 or loss.item() > 1e5:
            return True
        return False

    def mitigate_hindrance(self):
        """
        Mitigate training hindrances by reducing learning rate and resetting momentum.
        """
        self.lr *= 0.5
        self.m = [torch.zeros_like(param) for param in self.model.parameters()]
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def adapt_beta(self, grad):
        """
        Adapt momentum factor based on gradient variance.

        Args:
            grad: List of gradients for each parameter
        """
        grad_flat = torch.cat([g.view(-1) for g in grad])
        variance = torch.var(grad_flat)
        if variance > 1.0:
            self.beta = min(self.beta + 0.1, 0.99)
        else:
            self.beta = max(self.beta - 0.01, 0.8)

    def step(self, loss_fn, inputs, targets):
        """
        Perform one optimization step.

        Args:
            loss_fn: Loss function
            inputs: Input data
            targets: Target data

        Returns:
            float: Current loss value
        """
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        grad = [param.grad for param in self.model.parameters()]

        # Hindrance Detection and Mitigation
        if self.detect_hindrance(grad, loss):
            self.mitigate_hindrance()

        # Update momentum
        self.adapt_beta(grad)
        for i, param in enumerate(self.model.parameters()):
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * grad[i]

        # Update learning rate
        eta_t = self.lr * (1 + math.cos(math.pi * (self.epoch % self.T) / self.T)) / 2
        grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
        eta_t = eta_t / (1 + self.alpha * grad_norm**2)

        # Update parameters
        for i, param in enumerate(self.model.parameters()):
            param.data = param.data - eta_t * self.m[i]

        self.epoch += 1
        return loss.item()

    def train_epoch(self, data_loader, loss_fn):
        """
        Train for one epoch.

        Args:
            data_loader: DataLoader for training data
            loss_fn: Loss function

        Returns:
            float: Average loss for the epoch
        """
        total_loss = 0
        count = 0
        for batch in data_loader:
            inputs, targets = batch
            loss = self.step(loss_fn, inputs, targets)
            total_loss += loss
            count += 1
        return total_loss / count

    def train(self, data_loader, loss_fn, max_epochs=20, verbose=True):
        """
        Train the model for multiple epochs.

        Args:
            data_loader: DataLoader for training data
            loss_fn: Loss function
            max_epochs (int): Maximum number of epochs
            verbose (bool): Whether to print progress

        Returns:
            nn.Module: Trained model
        """
        for epoch in range(max_epochs):
            avg_loss = self.train_epoch(data_loader, loss_fn)
            if verbose:
                print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Learning Rate: {self.lr:.6f}")
        return self.model
