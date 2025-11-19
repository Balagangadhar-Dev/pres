"""
Neuroimaging Data Mining Research Toolkit

A comprehensive Python toolkit addressing key challenges in neuroimaging data mining:
- High dimensionality reduction
- Multi-modal data fusion
- Noise and artifact detection
- Interpretable AI models
- Privacy-preserving techniques
- Visualization tools

Author: AI Research Assistant
"""

from .data_generator import NeuroimagingDataGenerator
from .dimensionality import DimensionalityReducer
from .preprocessing import NoiseDetector, ArtifactRemover
from .interpretable import InterpretableClassifier
from .privacy import PrivacyPreserver
from .visualization import NeuroimagingVisualizer
from .multimodal import MultiModalFusion

__version__ = "1.0.0"
__all__ = [
    'NeuroimagingDataGenerator',
    'DimensionalityReducer',
    'NoiseDetector',
    'ArtifactRemover',
    'InterpretableClassifier',
    'PrivacyPreserver',
    'NeuroimagingVisualizer',
    'MultiModalFusion'
]
