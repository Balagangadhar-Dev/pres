#!/usr/bin/env python3
"""
Neuroimaging Data Mining Research Toolkit - Demonstration Script

This script demonstrates the comprehensive toolkit for addressing
key challenges in neuroimaging data mining research.

Run this script to see all modules in action with synthetic data.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import the toolkit
from neuroimaging_toolkit import (
    NeuroimagingDataGenerator,
    DimensionalityReducer,
    NoiseDetector,
    ArtifactRemover,
    InterpretableClassifier,
    PrivacyPreserver,
    NeuroimagingVisualizer,
    MultiModalFusion
)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def demo_data_generation():
    """Demonstrate synthetic data generation."""
    print_header("1. SYNTHETIC DATA GENERATION")
    print("Addressing Challenge #4: Small Sample Sizes\n")

    generator = NeuroimagingDataGenerator(random_state=42)

    # Generate structural MRI data
    print("Generating synthetic structural MRI data...")
    mri_data, mri_labels = generator.generate_structural_mri(
        n_samples=100,
        volume_shape=(32, 32, 32),
        n_classes=2,
        noise_level=0.1
    )
    print(f"  - Generated {len(mri_data)} 3D volumes")
    print(f"  - Volume shape: {mri_data[0].shape}")
    print(f"  - Classes: {np.unique(mri_labels)}")

    # Generate fMRI time series
    print("\nGenerating synthetic fMRI time series...")
    fmri_data, fmri_labels = generator.generate_fmri_timeseries(
        n_samples=50,
        n_timepoints=200,
        n_regions=100,
        n_classes=2
    )
    print(f"  - Generated {len(fmri_data)} time series")
    print(f"  - Shape: {fmri_data[0].shape} (timepoints x regions)")

    # Generate EEG data
    print("\nGenerating synthetic EEG data...")
    eeg_data, eeg_labels = generator.generate_eeg_data(
        n_samples=100,
        n_channels=64,
        n_timepoints=1000,
        n_classes=2
    )
    print(f"  - Generated {len(eeg_data)} EEG recordings")
    print(f"  - Shape: {eeg_data[0].shape} (channels x timepoints)")

    # Generate multi-modal data
    print("\nGenerating multi-modal dataset...")
    multimodal = generator.generate_multimodal_data(
        n_samples=100,
        include_mri=True,
        include_fmri=True,
        include_clinical=True,
        include_genetic=True
    )
    print("  Generated modalities:")
    for key, value in multimodal.items():
        if key != 'labels':
            print(f"    - {key}: {value.shape}")

    return mri_data, mri_labels, fmri_data, fmri_labels, multimodal


def demo_dimensionality_reduction(mri_data, mri_labels):
    """Demonstrate dimensionality reduction techniques."""
    print_header("2. DIMENSIONALITY REDUCTION")
    print("Addressing Challenge #1: High Dimensionality\n")

    # Flatten 3D volumes to feature vectors
    X = mri_data.reshape(len(mri_data), -1)
    print(f"Original data shape: {X.shape}")
    print(f"Features per sample: {X.shape[1]}")

    reducer = DimensionalityReducer(random_state=42)

    # PCA
    print("\n--- PCA Reduction ---")
    pca_data, pca_info = reducer.pca_reduction(X, variance_threshold=0.95)
    print(f"Reduced to {pca_info['n_components']} components")
    print(f"Variance explained: {pca_info['total_variance_explained']:.2%}")
    print(f"Reduction ratio: {pca_info['reduction_ratio']:.1f}x")

    # ICA
    print("\n--- ICA Reduction ---")
    ica_data, ica_info = reducer.ica_reduction(X, n_components=50)
    print(f"Extracted {ica_info['n_components']} independent components")

    # Feature Selection
    print("\n--- Feature Selection ---")
    fs_data, fs_indices, fs_info = reducer.feature_selection(X, mri_labels, k=100)
    print(f"Selected {fs_info['n_selected']} most informative features")
    print(f"Top 5 feature indices: {fs_info['top_features'][:5]}")

    # t-SNE for visualization
    print("\n--- t-SNE Visualization ---")
    tsne_data, tsne_info = reducer.tsne_visualization(X, n_components=2)
    print(f"Embedded to {tsne_info['n_components']}D for visualization")
    print(f"KL divergence: {tsne_info['kl_divergence']:.2f}")

    return pca_data, tsne_data


def demo_preprocessing(fmri_data):
    """Demonstrate noise detection and preprocessing."""
    print_header("3. NOISE DETECTION & PREPROCESSING")
    print("Addressing Challenge #3: Noise and Artifacts\n")

    detector = NoiseDetector()
    remover = ArtifactRemover()

    # Use first sample for demonstration
    sample = fmri_data[0]

    # Detect motion artifacts
    print("--- Motion Artifact Detection ---")
    motion_info = detector.detect_motion_artifacts(sample, threshold=3.0)
    print(f"Detected {motion_info['n_artifacts']} potential motion artifacts")
    print(f"Percentage affected: {motion_info['percentage_affected']:.1f}%")
    print(f"Mean framewise displacement: {motion_info['mean_fd']:.4f}")

    # Detect spikes
    print("\n--- Spike Detection ---")
    spike_info = detector.detect_spike_noise(sample, threshold=4.0)
    print(f"Detected {spike_info['n_spikes']} spike artifacts")

    # Estimate SNR
    print("\n--- SNR Estimation ---")
    snr = detector.estimate_snr(sample)
    print(f"Estimated SNR: {snr:.2f}")

    # Apply preprocessing pipeline
    print("\n--- Preprocessing Pipeline ---")
    preprocessed, preproc_info = remover.preprocess_pipeline(
        sample,
        sampling_rate=1.0,
        apply_bandpass=True,
        apply_despiking=True,
        apply_denoising=True
    )
    print(f"Applied steps: {preproc_info['steps_applied']}")
    print(f"Output shape: {preproc_info['output_shape']}")

    return preprocessed


def demo_interpretable_models(mri_data, mri_labels):
    """Demonstrate interpretable machine learning."""
    print_header("4. INTERPRETABLE AI MODELS")
    print("Addressing Challenge #5: Model Interpretability\n")

    # Flatten data
    X = mri_data.reshape(len(mri_data), -1)

    # Create feature names
    feature_names = [f'voxel_{i}' for i in range(X.shape[1])]

    classifier = InterpretableClassifier(random_state=42)

    # Logistic Regression
    print("--- L1 Logistic Regression (Sparse) ---")
    lr_results = classifier.train_logistic_regression(
        X, mri_labels,
        penalty='l1',
        C=0.1,
        feature_names=feature_names
    )
    print(f"CV Accuracy: {lr_results['cv_accuracy']:.3f}")
    print(f"Non-zero features: {lr_results['n_nonzero_features']}/{X.shape[1]}")
    print("Top 5 predictive features:")
    for name, imp, coef in lr_results['top_features'][:5]:
        direction = "+" if coef > 0 else "-"
        print(f"  {direction} {name}: {imp:.4f}")

    # Decision Tree
    print("\n--- Decision Tree (Rule-based) ---")
    dt_results = classifier.train_decision_tree(
        X, mri_labels,
        max_depth=4,
        feature_names=feature_names
    )
    print(f"CV Accuracy: {dt_results['cv_accuracy']:.3f}")
    print(f"Tree depth: {dt_results['tree_depth']}, Leaves: {dt_results['n_leaves']}")
    print("Sample decision rules:")
    for rule in dt_results['decision_rules'][:3]:
        print(f"  {rule[:80]}...")

    # Random Forest
    print("\n--- Random Forest (Ensemble) ---")
    rf_results = classifier.train_random_forest(
        X, mri_labels,
        n_estimators=100,
        feature_names=feature_names
    )
    print(f"CV Accuracy: {rf_results['cv_accuracy']:.3f}")
    print("Top 5 features (Gini importance):")
    for name, imp in rf_results['top_features_gini'][:5]:
        print(f"  {name}: {imp:.4f}")

    return classifier


def demo_privacy_preserving(mri_data, mri_labels):
    """Demonstrate privacy-preserving techniques."""
    print_header("5. PRIVACY-PRESERVING TECHNIQUES")
    print("Addressing Challenge #10: Ethical and Privacy Issues\n")

    # Flatten data
    X = mri_data.reshape(len(mri_data), -1)

    preserver = PrivacyPreserver(random_state=42)

    # Differential Privacy
    print("--- Differential Privacy Noise Addition ---")
    for epsilon in [0.1, 1.0, 10.0]:
        private_data, dp_info = preserver.add_differential_privacy(
            X, epsilon=epsilon, mechanism='laplace'
        )
        print(f"ε = {epsilon}: Noise magnitude = {dp_info['noise_magnitude']:.4f}")

    # Private Mean
    print("\n--- Differentially Private Mean ---")
    private_mean, mean_info = preserver.private_mean(X, epsilon=1.0)
    print(f"Privacy budget: ε = {mean_info['epsilon']}")
    print(f"Mean error: {mean_info['error']:.6f}")

    # Private Histogram
    print("\n--- Differentially Private Histogram ---")
    sample_feature = X[:, 0]
    private_counts, edges, hist_info = preserver.private_histogram(
        sample_feature, n_bins=10, epsilon=1.0
    )
    print(f"Total count error: {hist_info['total_error']:.2f}")

    # Federated Learning Simulation
    print("\n--- Federated Learning Simulation ---")
    fed_model, fed_info = preserver.simulate_federated_learning(
        X, mri_labels,
        n_sites=5,
        n_rounds=10,
        local_epochs=3
    )
    print(f"Number of sites: {fed_info['n_sites']}")
    print(f"Communication rounds: {fed_info['n_rounds']}")
    print(f"Final accuracy: {fed_info['final_accuracy']:.3f}")

    # Data Synthesis
    print("\n--- Synthetic Data Generation (Privacy-preserving) ---")
    synthetic, synth_info = preserver.data_synthesis(
        X[:50],  # Use subset
        n_synthetic=50,
        method='gaussian'
    )
    print(f"Generated {synth_info['n_synthetic']} synthetic samples")
    print(f"Utility score: {synth_info['utility_score']:.3f}")

    return preserver


def demo_multimodal_fusion(multimodal_data):
    """Demonstrate multi-modal fusion techniques."""
    print_header("6. MULTI-MODAL FUSION")
    print("Addressing Challenges #2 and #8: Data Heterogeneity and Integration\n")

    # Prepare modalities
    modalities = {
        'mri': multimodal_data['mri_features'],
        'fmri': multimodal_data['fmri_connectivity'][:, :500],  # Limit size
        'clinical': multimodal_data['clinical'],
        'genetic': multimodal_data['genetic'].astype(float)
    }
    labels = multimodal_data['labels']

    fusion = MultiModalFusion(random_state=42)

    # Early Fusion
    print("--- Early Fusion (Feature Concatenation) ---")
    early_fused, early_info = fusion.early_fusion(modalities, labels)
    print(f"Total features: {early_info['total_features']}")
    for name, info in early_info['modality_info'].items():
        print(f"  {name}: {info['n_features']} features")

    # Intermediate Fusion
    print("\n--- Intermediate Fusion (Shared Representation) ---")
    inter_fused, inter_info = fusion.intermediate_fusion(
        modalities, labels, n_components_per_modality=20
    )
    print(f"Total components: {inter_info['total_components']}")
    print("Variance explained per modality:")
    for name, var in inter_info['modality_variance_explained'].items():
        print(f"  {name}: {var:.2%}")

    # Attention Fusion
    print("\n--- Attention-based Fusion ---")
    attn_fused, attn_info = fusion.attention_fusion(modalities, labels)
    print("Learned attention weights:")
    for name, weight in attn_info['attention_weights'].items():
        print(f"  {name}: {weight:.3f}")

    # Compare all methods
    print("\n--- Fusion Method Comparison ---")
    comparison = fusion.compare_fusion_methods(modalities, labels)
    print("\nResults:")
    results_sorted = sorted(
        [(k, v) for k, v in comparison.items() if 'accuracy' in v],
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    for method, result in results_sorted[:6]:
        print(f"  {method}: {result['accuracy']:.3f} (+/- {result.get('std', 0):.3f})")

    return fusion


def demo_visualization(mri_data, tsne_data, mri_labels):
    """Demonstrate visualization capabilities."""
    print_header("7. VISUALIZATION")
    print("Creating visualizations for neuroimaging analysis\n")

    visualizer = NeuroimagingVisualizer()

    if visualizer.plt is None:
        print("Matplotlib not available. Skipping visualization demos.")
        print("Install with: pip install matplotlib")
        return

    print("Available visualization functions:")
    print("  - plot_brain_slices(): Visualize 3D brain volumes")
    print("  - plot_timeseries(): Plot fMRI/EEG time series")
    print("  - plot_connectivity_matrix(): Brain connectivity heatmap")
    print("  - plot_dimensionality_reduction(): t-SNE/PCA embeddings")
    print("  - plot_feature_importance(): Model interpretations")
    print("  - plot_training_history(): Training curves")
    print("  - plot_power_spectrum(): Frequency analysis")
    print("  - plot_privacy_utility_tradeoff(): Privacy vs accuracy")

    # Example: Create a summary
    print("\nTo create visualizations, call methods like:")
    print("  visualizer.plot_brain_slices(volume)")
    print("  visualizer.plot_dimensionality_reduction(tsne_data, labels)")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#    NEUROIMAGING DATA MINING RESEARCH TOOLKIT DEMONSTRATION    #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print("\nThis toolkit addresses the 10 key challenges in neuroimaging data mining:")
    print("  1. High Dimensionality of Data")
    print("  2. Heterogeneity of Data Sources")
    print("  3. Noise and Artifacts")
    print("  4. Small Sample Sizes")
    print("  5. Interpretability of Models")
    print("  6. Temporal and Spatial Complexity")
    print("  7. Standardization and Reproducibility")
    print("  8. Integration with Clinical/Genetic Data")
    print("  9. Computational Constraints")
    print("  10. Ethical and Privacy Issues")

    # Run demonstrations
    mri_data, mri_labels, fmri_data, fmri_labels, multimodal = demo_data_generation()
    pca_data, tsne_data = demo_dimensionality_reduction(mri_data, mri_labels)
    preprocessed = demo_preprocessing(fmri_data)
    classifier = demo_interpretable_models(mri_data, mri_labels)
    preserver = demo_privacy_preserving(mri_data, mri_labels)
    fusion = demo_multimodal_fusion(multimodal)
    demo_visualization(mri_data, tsne_data, mri_labels)

    # Summary
    print_header("DEMONSTRATION COMPLETE")
    print("This toolkit provides a comprehensive foundation for:")
    print("  - Synthetic data generation for algorithm development")
    print("  - Multiple dimensionality reduction techniques")
    print("  - Noise detection and artifact removal")
    print("  - Interpretable machine learning models")
    print("  - Privacy-preserving data analysis")
    print("  - Multi-modal data fusion")
    print("  - Comprehensive visualization")

    print("\nFor more information, see the README.md file.")
    print("\nNext steps:")
    print("  1. Apply to your own neuroimaging data")
    print("  2. Customize parameters for your specific use case")
    print("  3. Extend with additional algorithms")
    print("  4. Contribute improvements to the community")


if __name__ == "__main__":
    main()
