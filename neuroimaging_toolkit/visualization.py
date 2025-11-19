"""
Visualization Module for Neuroimaging Data

Provides comprehensive visualization tools for brain imaging data,
dimensionality reduction results, and model interpretations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings


class NeuroimagingVisualizer:
    """
    Visualization toolkit for neuroimaging data and analysis results.
    """

    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        self.style = style
        self._check_matplotlib()

    def _check_matplotlib(self):
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            if self.style != 'default':
                plt.style.use(self.style)
        except ImportError:
            warnings.warn("Matplotlib not available. Visualization functions will not work.")
            self.plt = None

    def plot_brain_slices(
        self,
        volume: np.ndarray,
        slices: Optional[List[int]] = None,
        title: str = "Brain Volume Slices",
        cmap: str = 'gray',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot axial, coronal, and sagittal slices of a 3D brain volume.

        Args:
            volume: 3D brain volume
            slices: Slice indices for each view [axial, coronal, sagittal]
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        if slices is None:
            slices = [s // 2 for s in volume.shape]

        fig, axes = self.plt.subplots(1, 3, figsize=(15, 5))

        # Axial (top-down)
        axes[0].imshow(volume[:, :, slices[2]], cmap=cmap, aspect='auto')
        axes[0].set_title(f'Axial (z={slices[2]})')
        axes[0].axis('off')

        # Coronal (front view)
        axes[1].imshow(volume[:, slices[1], :], cmap=cmap, aspect='auto')
        axes[1].set_title(f'Coronal (y={slices[1]})')
        axes[1].axis('off')

        # Sagittal (side view)
        axes[2].imshow(volume[slices[0], :, :], cmap=cmap, aspect='auto')
        axes[2].set_title(f'Sagittal (x={slices[0]})')
        axes[2].axis('off')

        fig.suptitle(title)
        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_timeseries(
        self,
        timeseries: np.ndarray,
        sampling_rate: float = 1.0,
        title: str = "Time Series",
        labels: Optional[List[str]] = None,
        highlight_regions: Optional[List[Tuple[int, int]]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot time series data (e.g., fMRI, EEG).

        Args:
            timeseries: Time series data (n_timepoints,) or (n_timepoints, n_channels)
            sampling_rate: Sampling rate in Hz
            title: Plot title
            labels: Labels for each channel
            highlight_regions: List of (start, end) tuples to highlight
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        n_timepoints, n_channels = timeseries.shape
        time = np.arange(n_timepoints) / sampling_rate

        fig, ax = self.plt.subplots(figsize=(14, 6))

        # Plot each channel
        for i in range(min(n_channels, 10)):  # Limit to 10 channels
            offset = i * 3  # Offset for visibility
            label = labels[i] if labels else f'Channel {i}'
            ax.plot(time, timeseries[:, i] + offset, label=label, alpha=0.8)

        # Highlight regions (e.g., artifacts)
        if highlight_regions:
            for start, end in highlight_regions:
                ax.axvspan(start / sampling_rate, end / sampling_rate,
                          alpha=0.3, color='red', label='Artifact')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal (offset for visibility)')
        ax.set_title(title)
        if n_channels <= 10:
            ax.legend(loc='upper right', fontsize=8)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_connectivity_matrix(
        self,
        connectivity: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Connectivity Matrix",
        cmap: str = 'RdBu_r',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot brain connectivity matrix (e.g., from fMRI).

        Args:
            connectivity: Square connectivity matrix
            labels: Region labels
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        fig, ax = self.plt.subplots(figsize=(10, 8))

        # Plot matrix
        im = ax.imshow(connectivity, cmap=cmap, aspect='equal',
                       vmin=-np.max(np.abs(connectivity)),
                       vmax=np.max(np.abs(connectivity)))

        # Colorbar
        cbar = self.plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Connectivity Strength')

        # Labels
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)

        ax.set_title(title)
        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_dimensionality_reduction(
        self,
        embedded: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Dimensionality Reduction",
        method_name: str = "Embedding",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot 2D or 3D embedding from dimensionality reduction.

        Args:
            embedded: Embedded data (n_samples, 2 or 3)
            labels: Class labels for coloring
            title: Plot title
            method_name: Name of DR method
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        if embedded.shape[1] == 2:
            fig, ax = self.plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                                c=labels if labels is not None else 'blue',
                                cmap='tab10', alpha=0.6, s=50)
            ax.set_xlabel(f'{method_name} 1')
            ax.set_ylabel(f'{method_name} 2')

            if labels is not None:
                legend = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend)

        elif embedded.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = self.plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                                c=labels if labels is not None else 'blue',
                                cmap='tab10', alpha=0.6, s=50)
            ax.set_xlabel(f'{method_name} 1')
            ax.set_ylabel(f'{method_name} 2')
            ax.set_zlabel(f'{method_name} 3')

        ax.set_title(title)
        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance from interpretable models.

        Args:
            importance: Feature importance values
            feature_names: Names of features
            top_k: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance))]

        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:top_k]
        sorted_importance = importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        fig, ax = self.plt.subplots(figsize=(10, 8))

        colors = self.plt.cm.viridis(sorted_importance / max(sorted_importance))
        bars = ax.barh(range(top_k), sorted_importance, color=colors)

        ax.set_yticks(range(top_k))
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training metrics over epochs.

        Args:
            history: Dictionary with metric names and lists of values
            title: Plot title
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        n_metrics = len(history)
        fig, axes = self.plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, history.items()):
            ax.plot(values, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.grid(True, alpha=0.3)

        fig.suptitle(title)
        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_power_spectrum(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        title: str = "Power Spectrum",
        highlight_bands: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot power spectrum from EEG/MEG data.

        Args:
            frequencies: Frequency values
            power: Power values
            title: Plot title
            highlight_bands: Whether to highlight standard EEG bands
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        fig, ax = self.plt.subplots(figsize=(12, 6))

        ax.semilogy(frequencies, power, linewidth=2, color='black')

        if highlight_bands:
            bands = {
                'Delta': (0.5, 4, 'red'),
                'Theta': (4, 8, 'orange'),
                'Alpha': (8, 13, 'green'),
                'Beta': (13, 30, 'blue'),
                'Gamma': (30, 100, 'purple')
            }

            for name, (low, high, color) in bands.items():
                mask = (frequencies >= low) & (frequencies <= high)
                if np.any(mask):
                    ax.fill_between(frequencies[mask], 0, power[mask],
                                   alpha=0.3, color=color, label=name)

            ax.legend(loc='upper right')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(title)
        ax.set_xlim([0, min(100, max(frequencies))])
        ax.grid(True, alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_class_comparison(
        self,
        data_class0: np.ndarray,
        data_class1: np.ndarray,
        feature_idx: int = 0,
        feature_name: str = "Feature",
        title: str = "Class Comparison",
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare feature distributions between classes.

        Args:
            data_class0: Data for class 0
            data_class1: Data for class 1
            feature_idx: Index of feature to compare
            feature_name: Name of the feature
            title: Plot title
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        fig, axes = self.plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(data_class0[:, feature_idx], bins=30, alpha=0.5,
                    label='Class 0', density=True)
        axes[0].hist(data_class1[:, feature_idx], bins=30, alpha=0.5,
                    label='Class 1', density=True)
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].set_title('Distribution')

        # Box plot
        axes[1].boxplot([data_class0[:, feature_idx], data_class1[:, feature_idx]],
                       labels=['Class 0', 'Class 1'])
        axes[1].set_ylabel(feature_name)
        axes[1].set_title('Box Plot')

        fig.suptitle(title)
        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def plot_privacy_utility_tradeoff(
        self,
        epsilons: List[float],
        accuracies: List[float],
        title: str = "Privacy-Utility Tradeoff",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot privacy vs utility tradeoff curve.

        Args:
            epsilons: Privacy budget values
            accuracies: Corresponding model accuracies
            title: Plot title
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        fig, ax = self.plt.subplots(figsize=(10, 6))

        ax.plot(epsilons, accuracies, 'bo-', linewidth=2, markersize=8)

        ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax.set_ylabel('Model Accuracy', fontsize=12)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Annotate
        ax.annotate('More Private', xy=(min(epsilons), min(accuracies)),
                   fontsize=10, color='red')
        ax.annotate('Less Private', xy=(max(epsilons), max(accuracies)),
                   fontsize=10, color='green')

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()

    def create_summary_figure(
        self,
        data: Dict[str, Any],
        title: str = "Neuroimaging Analysis Summary",
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive summary figure.

        Args:
            data: Dictionary containing various analysis results
            title: Overall figure title
            save_path: Path to save figure
        """
        if self.plt is None:
            print("Matplotlib not available")
            return

        fig = self.plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Sample brain slice (if available)
        if 'volume' in data:
            ax1 = fig.add_subplot(gs[0, 0])
            volume = data['volume']
            ax1.imshow(volume[:, :, volume.shape[2] // 2], cmap='gray')
            ax1.set_title('Sample Slice')
            ax1.axis('off')

        # Plot 2: Dimensionality reduction
        if 'embedded' in data:
            ax2 = fig.add_subplot(gs[0, 1])
            embedded = data['embedded']
            labels = data.get('labels')
            ax2.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap='tab10', alpha=0.6)
            ax2.set_title('t-SNE Embedding')

        # Plot 3: Feature importance
        if 'importance' in data:
            ax3 = fig.add_subplot(gs[0, 2])
            importance = data['importance'][:10]
            ax3.barh(range(10), importance)
            ax3.set_title('Top Features')

        # Plot 4: Training curve
        if 'loss_history' in data:
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.plot(data['loss_history'])
            ax4.set_title('Training Loss')
            ax4.set_xlabel('Epoch')

        # Plot 5: Model accuracy
        if 'accuracy' in data:
            ax5 = fig.add_subplot(gs[1, 1])
            acc = data['accuracy']
            ax5.bar(['Accuracy'], [acc], color='green')
            ax5.set_ylim([0, 1])
            ax5.set_title(f'Model Accuracy: {acc:.2%}')

        # Plot 6: Privacy budget
        if 'epsilon' in data:
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.pie([data['epsilon'], 10 - data['epsilon']],
                   labels=['Used', 'Remaining'],
                   autopct='%1.1f%%')
            ax6.set_title('Privacy Budget')

        fig.suptitle(title, fontsize=14, fontweight='bold')

        if save_path:
            self.plt.savefig(save_path, dpi=150, bbox_inches='tight')
        self.plt.show()
