# Neuroimaging Data Mining Research Toolkit

A comprehensive Python toolkit addressing the key challenges in neuroimaging data mining research. This toolkit provides practical implementations for researchers and practitioners working with brain imaging data.

## Overview

Based on the research challenges outlined in neuroimaging data mining, this toolkit provides solutions for:

1. **High Dimensionality of Data** - PCA, ICA, autoencoders, feature selection
2. **Heterogeneity of Data Sources** - Multi-modal fusion techniques
3. **Noise and Artifacts** - Detection and removal pipelines
4. **Small Sample Sizes** - Synthetic data generation
5. **Interpretability of Models** - Explainable AI methods
6. **Temporal and Spatial Complexity** - Time series analysis tools
7. **Standardization and Reproducibility** - Consistent pipelines
8. **Integration with Clinical/Genetic Data** - Multi-modal fusion
9. **Computational Constraints** - Efficient implementations
10. **Ethical and Privacy Issues** - Differential privacy, federated learning

## Installation

### Basic Installation

```bash
# Clone or download the repository
cd pres

# Install required dependencies
pip install -r requirements.txt
```

### Full Installation (with all optional dependencies)

```bash
pip install numpy scipy scikit-learn matplotlib torch shap
```

## Quick Start

### Run the Demo

```bash
python demo_neuroimaging.py
```

This will demonstrate all toolkit capabilities with synthetic data.

### Basic Usage

```python
from neuroimaging_toolkit import (
    NeuroimagingDataGenerator,
    DimensionalityReducer,
    InterpretableClassifier,
    PrivacyPreserver,
    MultiModalFusion
)

# Generate synthetic data
generator = NeuroimagingDataGenerator(random_state=42)
data, labels = generator.generate_structural_mri(n_samples=100)

# Flatten 3D volumes to features
X = data.reshape(len(data), -1)

# Reduce dimensionality
reducer = DimensionalityReducer()
X_reduced, info = reducer.pca_reduction(X, variance_threshold=0.95)
print(f"Reduced from {X.shape[1]} to {info['n_components']} features")

# Train interpretable model
classifier = InterpretableClassifier()
results = classifier.train_random_forest(X_reduced, labels)
print(f"Accuracy: {results['cv_accuracy']:.3f}")
```

## Module Documentation

### 1. Data Generator (`data_generator.py`)

Generate synthetic neuroimaging data for algorithm development and testing.

```python
from neuroimaging_toolkit import NeuroimagingDataGenerator

generator = NeuroimagingDataGenerator(random_state=42)

# Structural MRI (3D volumes)
mri_data, labels = generator.generate_structural_mri(
    n_samples=100,
    volume_shape=(64, 64, 64),
    n_classes=2
)

# fMRI time series
fmri_data, labels = generator.generate_fmri_timeseries(
    n_samples=50,
    n_timepoints=200,
    n_regions=100
)

# EEG data
eeg_data, labels = generator.generate_eeg_data(
    n_samples=100,
    n_channels=64,
    n_timepoints=1000
)

# Multi-modal dataset
multimodal = generator.generate_multimodal_data(
    n_samples=100,
    include_mri=True,
    include_fmri=True,
    include_clinical=True,
    include_genetic=True
)
```

### 2. Dimensionality Reduction (`dimensionality.py`)

Multiple techniques for reducing high-dimensional neuroimaging data.

```python
from neuroimaging_toolkit import DimensionalityReducer

reducer = DimensionalityReducer(random_state=42)

# PCA with variance threshold
X_pca, info = reducer.pca_reduction(X, variance_threshold=0.95)

# ICA for source separation
X_ica, info = reducer.ica_reduction(X, n_components=50)

# Feature selection based on labels
X_selected, indices, info = reducer.feature_selection(X, y, k=100)

# t-SNE for visualization
X_tsne, info = reducer.tsne_visualization(X, n_components=2)

# Deep autoencoder (requires PyTorch)
X_encoded, info = reducer.autoencoder_reduction(X, encoding_dim=32)
```

### 3. Preprocessing (`preprocessing.py`)

Detect and remove noise and artifacts from brain imaging data.

```python
from neuroimaging_toolkit import NoiseDetector, ArtifactRemover

# Detection
detector = NoiseDetector()
motion_info = detector.detect_motion_artifacts(timeseries)
spike_info = detector.detect_spike_noise(timeseries)
snr = detector.estimate_snr(volume)

# Removal
remover = ArtifactRemover()
filtered = remover.temporal_filtering(timeseries, low_freq=0.01, high_freq=0.1)
smoothed = remover.spatial_smoothing(volume, fwhm=6.0)
despiked = remover.spike_removal(timeseries)

# Complete pipeline
preprocessed, info = remover.preprocess_pipeline(
    timeseries,
    apply_bandpass=True,
    apply_despiking=True,
    apply_denoising=True
)
```

### 4. Interpretable AI (`interpretable.py`)

Explainable machine learning models for clinical applications.

```python
from neuroimaging_toolkit import InterpretableClassifier

classifier = InterpretableClassifier(random_state=42)

# Sparse logistic regression
lr_results = classifier.train_logistic_regression(X, y, penalty='l1')
print(f"Non-zero features: {lr_results['n_nonzero_features']}")

# Decision tree with rules
dt_results = classifier.train_decision_tree(X, y, max_depth=5)
print(dt_results['decision_rules'])

# Random forest with importance
rf_results = classifier.train_random_forest(X, y)
print(rf_results['top_features_gini'])

# Generate comprehensive report
report = classifier.generate_report(X, y, feature_names)
print(report)
```

### 5. Privacy Preservation (`privacy.py`)

Privacy-preserving techniques for sensitive neuroimaging data.

```python
from neuroimaging_toolkit import PrivacyPreserver

preserver = PrivacyPreserver(random_state=42)

# Differential privacy
private_data, info = preserver.add_differential_privacy(
    data, epsilon=1.0, mechanism='laplace'
)

# Private statistics
private_mean, info = preserver.private_mean(data, epsilon=1.0)
private_hist, edges, info = preserver.private_histogram(data, epsilon=1.0)

# Federated learning simulation
model, info = preserver.simulate_federated_learning(
    X, y,
    n_sites=5,
    n_rounds=10
)
print(f"Final accuracy: {info['final_accuracy']:.3f}")

# Synthetic data generation
synthetic, info = preserver.data_synthesis(data, method='gaussian')
```

### 6. Multi-Modal Fusion (`multimodal.py`)

Combine data from multiple imaging modalities and data sources.

```python
from neuroimaging_toolkit import MultiModalFusion

fusion = MultiModalFusion(random_state=42)

modalities = {
    'mri': mri_features,
    'fmri': fmri_connectivity,
    'clinical': clinical_data,
    'genetic': genetic_data
}

# Early fusion (concatenation)
fused, info = fusion.early_fusion(modalities, labels)

# Intermediate fusion (shared representation)
fused, info = fusion.intermediate_fusion(modalities, labels)

# Attention-based fusion
fused, info = fusion.attention_fusion(modalities, labels)
print(f"Attention weights: {info['attention_weights']}")

# Compare all methods
comparison = fusion.compare_fusion_methods(modalities, labels)
print(fusion.generate_fusion_report(modalities, labels))
```

### 7. Visualization (`visualization.py`)

Comprehensive visualization tools for neuroimaging analysis.

```python
from neuroimaging_toolkit import NeuroimagingVisualizer

viz = NeuroimagingVisualizer()

# Brain volume slices
viz.plot_brain_slices(volume, title="Brain MRI")

# Time series
viz.plot_timeseries(fmri_data, sampling_rate=2.0)

# Connectivity matrix
viz.plot_connectivity_matrix(correlation_matrix)

# Dimensionality reduction
viz.plot_dimensionality_reduction(tsne_embedding, labels)

# Feature importance
viz.plot_feature_importance(importance, feature_names)

# Power spectrum
viz.plot_power_spectrum(frequencies, power)

# Privacy-utility tradeoff
viz.plot_privacy_utility_tradeoff(epsilons, accuracies)
```

## Example Workflows

### Workflow 1: Complete Analysis Pipeline

```python
from neuroimaging_toolkit import *

# 1. Generate or load data
generator = NeuroimagingDataGenerator()
data, labels = generator.generate_structural_mri(n_samples=200)
X = data.reshape(len(data), -1)

# 2. Preprocess
remover = ArtifactRemover()
X_clean = remover.denoise_components(X)

# 3. Reduce dimensionality
reducer = DimensionalityReducer()
X_reduced, _ = reducer.pca_reduction(X_clean, variance_threshold=0.95)

# 4. Train interpretable model
classifier = InterpretableClassifier()
results = classifier.train_random_forest(X_reduced, labels)
print(f"Accuracy: {results['cv_accuracy']:.3f}")

# 5. Generate report
report = classifier.generate_report(X_reduced, labels)
```

### Workflow 2: Privacy-Preserving Multi-Site Analysis

```python
from neuroimaging_toolkit import PrivacyPreserver, InterpretableClassifier

# Simulate federated learning across hospitals
preserver = PrivacyPreserver()
model, fed_info = preserver.simulate_federated_learning(
    X, y,
    n_sites=10,
    n_rounds=20
)

# Evaluate with differential privacy
private_data, dp_info = preserver.add_differential_privacy(X, epsilon=1.0)
classifier = InterpretableClassifier()
results = classifier.train_logistic_regression(private_data, y)
print(f"Privacy budget: ε={dp_info['epsilon']}")
print(f"Accuracy: {results['cv_accuracy']:.3f}")
```

### Workflow 3: Multi-Modal Classification

```python
from neuroimaging_toolkit import *

# Generate multi-modal data
generator = NeuroimagingDataGenerator()
data = generator.generate_multimodal_data(n_samples=150)

# Prepare modalities
modalities = {
    'imaging': data['mri_features'],
    'functional': data['fmri_connectivity'][:, :500],
    'clinical': data['clinical']
}
labels = data['labels']

# Compare fusion methods
fusion = MultiModalFusion()
report = fusion.generate_fusion_report(modalities, labels)
print(report)
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional dimensionality reduction methods (UMAP, etc.)
- More sophisticated fusion architectures
- Support for real neuroimaging formats (NIfTI, DICOM)
- GPU acceleration for large datasets
- Additional privacy mechanisms

## References

This toolkit addresses challenges documented in:
- "Research Challenges in Data Mining in Neuroimaging"

Key techniques implemented are based on established methods in:
- Neuroimaging preprocessing (SPM, FSL, AFNI)
- Machine learning interpretability (SHAP, LIME)
- Differential privacy (Dwork et al.)
- Federated learning (McMahan et al.)

## License

This project is provided for research and educational purposes.

## Acknowledgments

Developed as a research toolkit to address the documented challenges in neuroimaging data mining. Designed to provide a starting point for researchers working with brain imaging data.
