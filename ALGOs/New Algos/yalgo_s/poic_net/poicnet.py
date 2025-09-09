import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import torchvision.transforms as transforms
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from typing import Optional, Union, List, Dict, Any, Tuple
import asyncio
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Exception Classes
class POICNetError(Exception):
    """Base exception class for POIC-Net errors."""
    pass

class ModelLoadError(POICNetError):
    """Raised when model loading fails."""
    pass

class ProcessingError(POICNetError):
    """Raised when processing fails."""
    pass

class DeviceError(POICNetError):
    """Raised when device-related errors occur."""
    pass

class InputValidationError(POICNetError):
    """Raised when input validation fails."""
    pass

class POICNet:
    """
    Partial Object Inference and Completion Network

    Enhanced with parallel processing and RTX 5060 optimizations for maximum performance.

    A multi-modal algorithm for detecting and completing partially visible objects
    using both visual and textual information.

    Args:
        image_model_name (str): Name of the image feature extraction model (default: "resnet50")
        text_model_name (str): Name of the text feature extraction model (default: "bert")
        threshold (float): Confidence threshold for detection (default: 0.5)
        device (str): Device to run on ('auto', 'cuda', 'cpu', 'mps')
        parallel_mode (str): Parallel processing mode ('none', 'thread', 'process', 'async')
        num_workers (int): Number of parallel workers (default: auto)
        use_rtx_optimizations (bool): Enable RTX 5060 specific optimizations
        batch_size (int): Batch size for processing (default: 16)
    """

    def __init__(self, image_model_name="resnet50", text_model_name="bert", threshold=0.5,
                 device='auto', parallel_mode='none', num_workers=None,
                 use_rtx_optimizations=True, batch_size=16):
        try:
            # Input validation
            self._validate_inputs(image_model_name, text_model_name, threshold, device, parallel_mode, batch_size)

            self.image_model_name = image_model_name
            self.text_model_name = text_model_name
            self.threshold = threshold
            self.batch_size = batch_size
            self.use_rtx_optimizations = use_rtx_optimizations

            # Device configuration with RTX 5060 optimizations
            self.device = self._configure_device(device)

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

            # Initialize models with device awareness and error handling
            try:
                self.image_model = self._load_image_model(image_model_name)
                self.text_tokenizer, self.text_model = self._load_text_model(text_model_name)
                logger.info(f"Models loaded: {image_model_name}, {text_model_name}")
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise ModelLoadError(f"Cannot load models: {e}")

            # Move models to device with error handling
            try:
                self.image_model = self.image_model.to(self.device)
                self.text_model = self.text_model.to(self.device)
                logger.info(f"Models moved to device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to move models to device {self.device}: {e}")
                raise DeviceError(f"Cannot move models to device {self.device}: {e}")

            # Image preprocessing with GPU acceleration
            self.image_transform = self._create_optimized_transforms()

            # Performance monitoring
            self.performance_stats = {
                'processing_time': [],
                'gpu_utilization': [],
                'memory_usage': [],
                'throughput': []
            }

            # Caching for repeated computations
            self._feature_cache = {}
            self._completion_cache = {}

            logger.info("POIC-Net initialized successfully")

        except Exception as e:
            logger.error(f"POIC-Net initialization failed: {e}")
            raise POICNetError(f"Failed to initialize POIC-Net: {e}") from e

    def _validate_inputs(self, image_model_name, text_model_name, threshold, device, parallel_mode, batch_size):
        """Validate input parameters."""
        valid_image_models = ['resnet50', 'resnet101', 'vgg16']
        if image_model_name not in valid_image_models:
            raise InputValidationError(f"Invalid image model: {image_model_name}. Must be one of {valid_image_models}")

        valid_text_models = ['bert', 'gpt2']
        if text_model_name not in valid_text_models:
            raise InputValidationError(f"Invalid text model: {text_model_name}. Must be one of {valid_text_models}")

        if not (0 < threshold <= 1):
            raise InputValidationError(f"Threshold must be between 0 and 1, got {threshold}")

        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if device not in valid_devices and not device.startswith(('cuda:', 'cpu')):
            raise DeviceError(f"Invalid device: {device}. Must be one of {valid_devices}")

        valid_parallel_modes = ['none', 'thread', 'process', 'async']
        if parallel_mode not in valid_parallel_modes:
            raise InputValidationError(f"Invalid parallel mode: {parallel_mode}. Must be one of {valid_parallel_modes}")

        if batch_size <= 0:
            raise InputValidationError(f"Batch size must be positive, got {batch_size}")

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
            return min(8, torch.cuda.device_count() * 4)  # More workers for GPU
        elif self.device.type == 'mps':
            return 4
        else:
            return max(2, mp.cpu_count() // 2)

    def _setup_parallel_executor(self):
        """Setup parallel execution environment."""
        if self.parallel_mode == 'thread':
            return ThreadPoolExecutor(max_workers=self.num_workers)
        elif self.parallel_mode == 'process':
            return ProcessPoolExecutor(max_workers=self.num_workers)
        elif self.parallel_mode == 'async':
            return None  # Will use asyncio
        else:
            return None

    def _create_optimized_transforms(self):
        """Create optimized image transforms with GPU acceleration."""
        if self.device.type == 'cuda':
            # Use GPU-accelerated transforms when available
            try:
                import torchvision.transforms.v2 as transforms_v2
                return transforms_v2.Compose([
                    transforms_v2.Resize((224, 224)),
                    transforms_v2.ToTensor(),
                    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms_v2.ToDtype(torch.float32)
                ])
            except ImportError:
                pass

        # Fallback to standard transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    async def _async_process_batch(self, batch_data: List[Any], process_func) -> List[Any]:
        """Process batch data asynchronously."""
        if self.parallel_mode == 'async':
            tasks = [process_func(item) for item in batch_data]
            return await asyncio.gather(*tasks)
        else:
            return [process_func(item) for item in batch_data]

    def _parallel_feature_extraction(self, images: List[Any]) -> List[torch.Tensor]:
        """Extract features from multiple images in parallel."""
        if self.parallel_mode == 'thread' and self.executor:
            futures = []
            for image in images:
                future = self.executor.submit(self.extract_image_features, image)
                futures.append(future)

            return [future.result() for future in futures]
        else:
            return [self.extract_image_features(image) for image in images]

    def _batch_process_images(self, images: List[Any]) -> torch.Tensor:
        """Process multiple images as a batch for better GPU utilization."""
        if len(images) == 1:
            return self.extract_image_features(images[0])

        # Convert images to tensors
        image_tensors = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                tensor = self.image_transform(image)
                image_tensors.append(tensor)

        # Batch processing
        if image_tensors:
            batch_tensor = torch.stack(image_tensors).to(self.device)

            with torch.no_grad():
                batch_features = self.image_model(batch_tensor)

            return batch_features

        return torch.empty(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()

        if self.device.type == 'cuda':
            stats['current_gpu_util'] = torch.cuda.utilization(self.device)
            stats['current_memory'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            stats['peak_memory'] = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB

        return stats

    def optimize_memory_usage(self):
        """Optimize memory usage for RTX 5060."""
        if self.device.type == 'cuda':
            # Clear cache
            torch.cuda.empty_cache()

            # Use mixed precision if available
            if hasattr(torch.cuda, 'amp') and self.use_rtx_optimizations:
                self.scaler = torch.cuda.amp.GradScaler()

    def enable_multi_gpu(self, gpu_ids: List[int] = None):
        """Enable multi-GPU processing for RTX 5060 setups."""
        if not torch.cuda.is_available():
            logging.warning("Multi-GPU requested but CUDA not available")
            return

        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))

        if len(gpu_ids) > 1:
            # Wrap model for multi-GPU
            self.image_model = nn.DataParallel(self.image_model, device_ids=gpu_ids)
            self.text_model = nn.DataParallel(self.text_model, device_ids=gpu_ids)
            logging.info(f"Enabled multi-GPU processing on GPUs: {gpu_ids}")
        else:
            logging.info("Single GPU mode - no multi-GPU setup needed")

    def _load_image_model(self, model_name):
        """Load pre-trained image model for feature extraction."""
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=True)
        elif model_name == "vgg16":
            model = models.vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported image model: {model_name}")

        model.eval()
        return model

    def _load_text_model(self, model_name):
        """Load pre-trained text model for feature extraction."""
        if model_name == "bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == "gpt2":
            from transformers import GPT2Tokenizer, GPT2Model
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2Model.from_pretrained('gpt2')
        else:
            raise ValueError(f"Unsupported text model: {model_name}")

        model.eval()
        return tokenizer, model

    def extract_image_features(self, image):
        """
        Extract features from image with comprehensive error handling.

        Args:
            image: PIL Image or tensor

        Returns:
            torch.Tensor: Image features

        Raises:
            ProcessingError: If feature extraction fails
            InputValidationError: If input is invalid
        """
        try:
            # Input validation
            if image is None:
                raise InputValidationError("Image cannot be None")

            # Convert to tensor if needed
            if not isinstance(image, torch.Tensor):
                try:
                    image = self.image_transform(image).unsqueeze(0)
                except Exception as e:
                    raise InputValidationError(f"Cannot process image: {e}")

            # Move to device
            image = image.to(self.device)

            # Extract features
            with torch.no_grad():
                try:
                    features = self.image_model(image)
                except Exception as e:
                    raise ProcessingError(f"Feature extraction failed: {e}")

            return features

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during image feature extraction")
            self.optimize_memory_usage()
            raise ProcessingError("CUDA out of memory - try reducing image size or batch size")

        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}")
            raise ProcessingError(f"Failed to extract image features: {e}") from e

    def extract_text_features(self, text):
        """
        Extract features from text with comprehensive error handling.

        Args:
            text (str): Input text

        Returns:
            torch.Tensor: Text features

        Raises:
            ProcessingError: If feature extraction fails
            InputValidationError: If input is invalid
        """
        try:
            # Input validation
            if not isinstance(text, str):
                raise InputValidationError(f"Text must be a string, got {type(text)}")

            if not text.strip():
                raise InputValidationError("Text cannot be empty")

            if len(text) > 512:  # BERT token limit
                logger.warning(f"Text length {len(text)} exceeds BERT limit, truncating")
                text = text[:512]

            # Tokenize text
            try:
                encoded_input = self.text_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            except Exception as e:
                raise ProcessingError(f"Text tokenization failed: {e}")

            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            # Extract features
            with torch.no_grad():
                try:
                    outputs = self.text_model(**encoded_input)
                    features = outputs.last_hidden_state.mean(dim=1)
                except Exception as e:
                    raise ProcessingError(f"Text feature extraction failed: {e}")

            return features

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during text feature extraction")
            self.optimize_memory_usage()
            raise ProcessingError("CUDA out of memory - try reducing text length")

        except Exception as e:
            logger.error(f"Text feature extraction failed: {e}")
            raise ProcessingError(f"Failed to extract text features: {e}") from e

    def detect_partial_objects(self, features):
        """
        Partial Object Detection Module (PODM).

        Args:
            features: Image features

        Returns:
            tuple: (partial_regions, confidence_scores)
        """
        # Simple threshold-based detection
        feature_map = features.mean(dim=1, keepdim=True)
        pooled = torch.nn.functional.adaptive_max_pool2d(feature_map, (1, 1))
        confidence = pooled.item()

        if confidence > self.threshold:
            h, w = features.shape[2:]
            x1, y1 = int(0.25 * w), int(0.25 * h)
            x2, y2 = int(0.75 * w), int(0.75 * h)
            return [[x1, y1, x2, y2]], [confidence]
        else:
            return [[]], [0.0]

    def complete_object(self, region, features):
        """
        Generative Completion Network (GCN).

        Args:
            region: Bounding box [x1, y1, x2, y2]
            features: Image features

        Returns:
            torch.Tensor: Completed object image
        """
        if not region:
            return torch.zeros(3, 224, 224)

        # Simple completion using upsampling
        upsampled = torch.nn.functional.interpolate(features, size=(224, 224), mode='bilinear', align_corners=False)
        completed = upsampled.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).squeeze(0)
        return completed

    def multimodal_attention(self, completed_objects, text_features):
        """
        Multi-Modal Attention Module (MMAM).

        Args:
            completed_objects: List of completed object tensors
            text_features: Text features

        Returns:
            list: Refined completed objects
        """
        # Placeholder: Return as is for now
        return completed_objects

    def agentic_feedback_loop(self, completed_objects):
        """
        Agentic Feedback Loop (AFL).

        Args:
            completed_objects: List of completed object tensors

        Returns:
            list: Refined completed objects
        """
        # Placeholder: Return as is for now
        return completed_objects

    def __call__(self, input_data, modality="image"):
        """
        Process input data through the POIC-Net pipeline with comprehensive error handling.

        Args:
            input_data: Image, text, or (image, text) tuple
            modality (str): "image", "text", or "multimodal"

        Returns:
            tuple: (refined_objects, confidence_scores)

        Raises:
            ProcessingError: If processing fails
            InputValidationError: If input is invalid
        """
        try:
            # Input validation
            if input_data is None:
                raise InputValidationError("Input data cannot be None")

            valid_modalities = ["image", "text", "multimodal"]
            if modality not in valid_modalities:
                raise InputValidationError(f"Invalid modality: {modality}. Must be one of {valid_modalities}")

            start_time = time.time()

            # Process input based on type
            if isinstance(input_data, tuple) and len(input_data) == 2:
                image_data, text_data = input_data
                try:
                    features = self.extract_image_features(image_data)
                    text_features = self.extract_text_features(text_data)
                    use_multimodal = True
                except Exception as e:
                    raise ProcessingError(f"Failed to process multimodal input: {e}")
            else:
                if modality == "image":
                    try:
                        features = self.extract_image_features(input_data)
                        text_features = None
                        use_multimodal = False
                    except Exception as e:
                        raise ProcessingError(f"Failed to process image input: {e}")
                else:
                    raise InputValidationError("Text-only modality not fully implemented")

            # Detect partial objects
            try:
                partial_regions, confidence_scores = self.detect_partial_objects(features)
            except Exception as e:
                raise ProcessingError(f"Partial object detection failed: {e}")

            # Complete objects
            completed_objects = []
            try:
                for i, region in enumerate(partial_regions):
                    if confidence_scores[i] > self.threshold:
                        completed_object = self.complete_object(region, features)
                        completed_objects.append(completed_object)
            except Exception as e:
                raise ProcessingError(f"Object completion failed: {e}")

            # Multi-modal attention
            if use_multimodal and text_features is not None:
                try:
                    completed_objects = self.multimodal_attention(completed_objects, text_features)
                except Exception as e:
                    logger.warning(f"Multi-modal attention failed: {e}")
                    warnings.warn("Multi-modal attention failed, using image-only results", UserWarning)

            # Agentic feedback loop
            try:
                refined_objects = self.agentic_feedback_loop(completed_objects)
            except Exception as e:
                logger.warning(f"Agentic feedback loop failed: {e}")
                warnings.warn("Agentic feedback failed, using completed objects", UserWarning)
                refined_objects = completed_objects

            # Performance monitoring
            processing_time = time.time() - start_time
            self.performance_stats['processing_time'].append(processing_time)

            logger.info(f"POIC-Net processing completed in {processing_time:.3f}s")
            return refined_objects, confidence_scores

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during POIC-Net processing")
            self.optimize_memory_usage()
            raise ProcessingError("CUDA out of memory - try reducing input size or batch size")

        except Exception as e:
            logger.error(f"POIC-Net processing failed: {e}")
            raise ProcessingError(f"POIC-Net processing failed: {e}") from e
