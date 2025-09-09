"""
YALGO-S: Yet Another Library for Gradient Optimization and Specialized algorithms

A collection of advanced algorithms for machine learning optimization and multi-modal processing.
"""

from .agmohd import AGMOHD
from .poic_net import POICNet
from .image_training import ImageTrainer
# from .arce import ARCE  # To be added when available

__version__ = "0.1.0"
__all__ = ["AGMOHD", "POICNet", "ImageTrainer"]
