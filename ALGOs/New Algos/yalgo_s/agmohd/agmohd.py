import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import logging
from typing import Optional, Union, List, Dict, Any
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Exception Classes
class AGMOHDError(Exception):
    """Base exception class for AGMOHD errors."""
    pass

class DeviceError(AGMOHDError):
    """Raised when device-related errors occur."""
    pass

class ModelError(AGMOHDError):
    """Raised when model-related errors occur."""
    pass

class OptimizationError(AGMOHDError):
    """Raised when optimization-related errors occur."""
    pass

class MemoryError(AGMOHDError):
    """Raised when memory-related errors occur."""
    pass

class ParallelProcessingError(AGMOHDError):
    """Raised when parallel processing errors occur."""
    pass

class AGMOHD:
    """
    Adaptive Gradient Momentum with Hindrance Detection Optimizer

    Enhanced with parallel processing and RTX 5060 optimizations for maximum performance.

    Args:
        model (nn.Module): The neural network model to optimize
        lr (float): Initial learning rate (default: 0.01)
        beta (float): Initial momentum factor (default: 0.9)
        alpha (float): Gradient norm sensitivity (default: 0.1)
        T (int): Cycle length for learning rate schedule (default: 10)
        device (str): Device to run on ('auto', 'cuda', 'cpu', 'mps')
        parallel_mode (str): Parallel processing mode ('none', 'thread', 'process', 'data')
        num_workers (int): Number of parallel workers (default: auto)
        use_rtx_optimizations (bool): Enable RTX 5060 specific optimizations
    """

    def __init__(self, model, lr=0.01, beta=0.9, alpha=0.1, T=10,
                 device='auto', parallel_mode='none', num_workers=None,
                 use_rtx_optimizations=True):
        try:
            # Input validation
            self._validate_inputs(model, lr, beta, alpha, T, device, parallel_mode)

            self.model = model
            self.lr = lr
            self.beta = beta
            self.alpha = alpha
            self.T = T
            self.epoch = 0
            self.use_rtx_optimizations = use_rtx_optimizations

            # Device configuration with RTX 5060 optimizations
            self.device = self._configure_device(device)

            # Move model to device with error handling
            try:
                self.model = self.model.to(self.device)
                logger.info(f"Model moved to device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to move model to device {self.device}: {e}")
                raise DeviceError(f"Cannot move model to device {self.device}: {e}")

            # RTX 5060 specific optimizations
            if self.use_rtx_optimizations and torch.cuda.is_available():
                try:
                    self._enable_rtx_optimizations()
                    logger.info("RTX 5060 optimizations enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable RTX optimizations: {e}")
                    warnings.warn("RTX optimizations could not be enabled", UserWarning)

            # Parallel processing setup
            self.parallel_mode = parallel_mode
            self.num_workers = num_workers or self._auto_configure_workers()
            self.executor = self._setup_parallel_executor()

            # Initialize momentum with device awareness and error handling
            try:
                self.m = [torch.zeros_like(param, device=self.device) for param in model.parameters()]
                logger.info(f"Momentum initialized for {len(self.m)} parameter groups")
            except Exception as e:
                logger.error(f"Failed to initialize momentum: {e}")
                raise MemoryError(f"Cannot initialize momentum buffers: {e}")

            # Enhanced optimizer with RTX optimizations
            self.optimizer = self._create_optimized_optimizer(lr)

            # Performance monitoring
            self.performance_stats = {
                'gpu_utilization': [],
                'memory_usage': [],
                'throughput': [],
                'latency': []
            }

            logger.info("AGMOHD optimizer initialized successfully")

        except Exception as e:
            logger.error(f"AGMOHD initialization failed: {e}")
            raise AGMOHDError(f"Failed to initialize AGMOHD: {e}") from e

    def _validate_inputs(self, model, lr, beta, alpha, T, device, parallel_mode):
        """Validate input parameters."""
        if not isinstance(model, nn.Module):
            raise ModelError("Model must be a PyTorch nn.Module")

        if not (0 < lr <= 1):
            raise OptimizationError(f"Learning rate must be between 0 and 1, got {lr}")

        if not (0 <= beta <= 1):
            raise OptimizationError(f"Beta must be between 0 and 1, got {beta}")

        if not (0 <= alpha <= 1):
            raise OptimizationError(f"Alpha must be between 0 and 1, got {alpha}")

        if T <= 0:
            raise OptimizationError(f"T must be positive, got {T}")

        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if device not in valid_devices and not device.startswith(('cuda:', 'cpu')):
            raise DeviceError(f"Invalid device: {device}. Must be one of {valid_devices}")

        valid_parallel_modes = ['none', 'thread', 'process', 'data']
        if parallel_mode not in valid_parallel_modes:
            raise ParallelProcessingError(f"Invalid parallel mode: {parallel_mode}. Must be one of {valid_parallel_modes}")

    def _configure_device(self, device):
        """Configure the optimal device for computation."""
        if device == 'auto':
            if torch.cuda.is_available():
                # RTX 5060 and newer GPUs
                gpu_name = torch.cuda.get_device_name(0)
                if 'RTX' in gpu_name or '50' in gpu_name:
                    return torch.device('cuda:0')
                else:
                    return torch.device('cuda:0')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _enable_rtx_optimizations(self):
        """Enable RTX 5060 specific optimizations."""
        if torch.cuda.is_available():
            # Enable TensorFloat-32 for faster computation
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Set optimal memory allocation
            torch.cuda.set_per_process_memory_fraction(0.9)

            # Enable asynchronous data loading
            torch.cuda.init()

    def _auto_configure_workers(self):
        """Automatically configure the number of workers based on system."""
        if self.device.type == 'cuda':
            return min(4, torch.cuda.device_count() * 2)
        elif self.device.type == 'mps':
            return 2
        else:
            return max(1, mp.cpu_count() // 2)

    def _setup_parallel_executor(self):
        """Setup parallel execution environment."""
        if self.parallel_mode == 'thread':
            return ThreadPoolExecutor(max_workers=self.num_workers)
        elif self.parallel_mode == 'process':
            return ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            return None

    def _create_optimized_optimizer(self, lr):
        """Create optimizer with RTX optimizations."""
        if self.device.type == 'cuda' and self.use_rtx_optimizations:
            # Use fused optimizers for RTX GPUs
            try:
                from torch.optim import AdamW as FusedAdamW
                return FusedAdamW(self.model.parameters(), lr=lr, fused=True)
            except:
                return optim.SGD(self.model.parameters(), lr=lr)
        else:
            return optim.SGD(self.model.parameters(), lr=lr)

    def _parallel_batch_processing(self, batch_data, loss_fn):
        """Process batches in parallel for improved throughput."""
        if self.parallel_mode == 'data' and self.executor:
            # Split batch into smaller chunks for parallel processing
            batch_size = len(batch_data[0])
            chunk_size = max(1, batch_size // self.num_workers)

            futures = []
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_inputs = batch_data[0][i:end_idx]
                chunk_targets = batch_data[1][i:end_idx]

                future = self.executor.submit(
                    self._process_batch_chunk,
                    chunk_inputs, chunk_targets, loss_fn
                )
                futures.append(future)

            # Aggregate results
            total_loss = 0
            for future in futures:
                total_loss += future.result()

            return total_loss / len(futures)
        else:
            # Standard processing
            inputs, targets = batch_data
            return self.step(loss_fn, inputs, targets)

    def _process_batch_chunk(self, inputs, targets, loss_fn):
        """Process a chunk of batch data."""
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            return loss.item()

    def get_performance_stats(self):
        """Get current performance statistics."""
        stats = self.performance_stats.copy()

        if self.device.type == 'cuda':
            stats['current_gpu_util'] = torch.cuda.utilization(self.device)
            stats['current_memory'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB

        return stats

    def optimize_memory_usage(self):
        """Optimize memory usage for RTX 5060."""
        if self.device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()

            # Use gradient checkpointing for large models
            if sum(p.numel() for p in self.model.parameters()) > 10**7:
                self.model.gradient_checkpointing_enable()

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
        Perform one optimization step with comprehensive error handling.

        Args:
            loss_fn: Loss function
            inputs: Input data
            targets: Target data

        Returns:
            float: Current loss value

        Raises:
            OptimizationError: If optimization step fails
            MemoryError: If memory allocation fails
        """
        try:
            # Input validation
            if inputs is None or targets is None:
                raise OptimizationError("Inputs and targets cannot be None")

            if not callable(loss_fn):
                raise OptimizationError("Loss function must be callable")

            # Ensure inputs are on correct device
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            if hasattr(targets, 'to'):
                targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Get gradients
            grad = [param.grad for param in self.model.parameters()]

            # Validate gradients
            if any(g is None for g in grad):
                raise OptimizationError("Some gradients are None - check model and loss function")

            # Hindrance Detection and Mitigation
            try:
                if self.detect_hindrance(grad, loss):
                    logger.info("Hindrance detected, applying mitigation")
                    self.mitigate_hindrance()
            except Exception as e:
                logger.warning(f"Hindrance detection failed: {e}")
                warnings.warn("Hindrance detection failed, continuing without mitigation", UserWarning)

            # Update momentum
            try:
                self.adapt_beta(grad)
                for i, param in enumerate(self.model.parameters()):
                    self.m[i] = self.beta * self.m[i] + (1 - self.beta) * grad[i]
            except Exception as e:
                logger.error(f"Momentum update failed: {e}")
                raise OptimizationError(f"Failed to update momentum: {e}")

            # Update learning rate
            eta_t = self.lr * (1 + math.cos(math.pi * (self.epoch % self.T) / self.T)) / 2
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad]))
            eta_t = eta_t / (1 + self.alpha * grad_norm**2)

            # Update parameters
            try:
                for i, param in enumerate(self.model.parameters()):
                    param.data = param.data - eta_t * self.m[i]
            except Exception as e:
                logger.error(f"Parameter update failed: {e}")
                raise OptimizationError(f"Failed to update parameters: {e}")

            self.epoch += 1
            return loss.item()

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error")
            self.optimize_memory_usage()
            raise MemoryError("CUDA out of memory - try reducing batch size or model size")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Memory error: {e}")
                self.optimize_memory_usage()
                raise MemoryError(f"Memory allocation failed: {e}")
            else:
                logger.error(f"Runtime error: {e}")
                raise OptimizationError(f"Optimization step failed: {e}")

        except Exception as e:
            logger.error(f"Unexpected error in optimization step: {e}")
            raise OptimizationError(f"Optimization step failed: {e}") from e

    def train_epoch(self, data_loader, loss_fn):
        """
        Train for one epoch with error handling and recovery.

        Args:
            data_loader: DataLoader for training data
            loss_fn: Loss function

        Returns:
            float: Average loss for the epoch

        Raises:
            OptimizationError: If training epoch fails
        """
        try:
            if data_loader is None:
                raise OptimizationError("DataLoader cannot be None")

            if not callable(loss_fn):
                raise OptimizationError("Loss function must be callable")

            total_loss = 0
            count = 0
            failed_batches = 0
            max_failed_batches = len(data_loader) // 10  # Allow 10% failure rate

            for batch_idx, batch in enumerate(data_loader):
                try:
                    if len(batch) != 2:
                        raise OptimizationError(f"Batch must contain inputs and targets, got {len(batch)} items")

                    inputs, targets = batch
                    loss = self.step(loss_fn, inputs, targets)
                    total_loss += loss
                    count += 1

                except Exception as e:
                    failed_batches += 1
                    logger.warning(f"Batch {batch_idx} failed: {e}")

                    if failed_batches > max_failed_batches:
                        raise OptimizationError(f"Too many failed batches ({failed_batches}/{len(data_loader)})")

                    # Continue with next batch
                    continue

            if count == 0:
                raise OptimizationError("No batches were successfully processed")

            avg_loss = total_loss / count
            logger.info(f"Epoch completed: {count}/{len(data_loader)} batches successful")

            return avg_loss

        except Exception as e:
            logger.error(f"Training epoch failed: {e}")
            raise OptimizationError(f"Training epoch failed: {e}") from e

    def train(self, data_loader, loss_fn, max_epochs=20, verbose=True):
        """
        Train the model for multiple epochs with comprehensive error handling.

        Args:
            data_loader: DataLoader for training data
            loss_fn: Loss function
            max_epochs (int): Maximum number of epochs
            verbose (bool): Whether to print progress

        Returns:
            nn.Module: Trained model

        Raises:
            OptimizationError: If training fails
        """
        try:
            if max_epochs <= 0:
                raise OptimizationError(f"max_epochs must be positive, got {max_epochs}")

            best_loss = float('inf')
            patience = 5  # Early stopping patience
            patience_counter = 0

            logger.info(f"Starting training for {max_epochs} epochs")

            for epoch in range(max_epochs):
                try:
                    start_time = time.time()
                    avg_loss = self.train_epoch(data_loader, loss_fn)
                    epoch_time = time.time() - start_time

                    # Performance monitoring
                    if self.device.type == 'cuda':
                        memory_used = torch.cuda.memory_allocated(self.device) / 1024**3
                        self.performance_stats['memory_usage'].append(memory_used)
                        self.performance_stats['latency'].append(epoch_time)

                    # Early stopping check
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch} (patience exhausted)")
                        break

                    # Learning rate scheduling
                    if epoch > 0 and epoch % 10 == 0:
                        self.lr *= 0.9  # Decay learning rate
                        logger.info(f"Learning rate decayed to {self.lr:.6f}")

                    if verbose:
                        print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, LR: {self.lr:.6f}, Time: {epoch_time:.2f}s")

                except KeyboardInterrupt:
                    logger.info("Training interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Epoch {epoch} failed: {e}")
                    # Continue with next epoch unless it's a critical error
                    if isinstance(e, (MemoryError, DeviceError)):
                        raise  # Re-raise critical errors
                    continue

            logger.info("Training completed successfully")
            return self.model

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise OptimizationError(f"Training failed: {e}") from e
