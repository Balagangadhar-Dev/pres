"""
Synthetic Neuroimaging Data Generator

Generates realistic synthetic neuroimaging data for research and demonstration.
Addresses Challenge #4: Small Sample Sizes
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter


class NeuroimagingDataGenerator:
    """
    Generate synthetic neuroimaging data resembling real brain scans.

    This helps address the challenge of small sample sizes by allowing
    researchers to test algorithms on synthetic data before applying
    to real datasets.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator.

        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def generate_structural_mri(
        self,
        n_samples: int = 100,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        n_classes: int = 2,
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic structural MRI-like 3D volumes.

        Args:
            n_samples: Number of samples to generate
            volume_shape: Shape of each 3D volume (x, y, z)
            n_classes: Number of classes (e.g., healthy vs diseased)
            noise_level: Amount of Gaussian noise to add

        Returns:
            Tuple of (data array, labels array)
        """
        data = []
        labels = []

        for i in range(n_samples):
            # Create base brain-like structure
            volume = self._create_brain_structure(volume_shape)

            # Assign class and add class-specific patterns
            label = i % n_classes
            if label == 1:
                # Add "abnormality" pattern for diseased class
                volume = self._add_lesion_pattern(volume)

            # Add realistic noise
            volume = self._add_noise(volume, noise_level)

            data.append(volume)
            labels.append(label)

        return np.array(data), np.array(labels)

    def generate_fmri_timeseries(
        self,
        n_samples: int = 50,
        n_timepoints: int = 200,
        n_regions: int = 100,
        n_classes: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic fMRI time series data.

        Addresses Challenge #6: Temporal and Spatial Complexity

        Args:
            n_samples: Number of subjects
            n_timepoints: Number of time points per scan
            n_regions: Number of brain regions (ROIs)
            n_classes: Number of classes

        Returns:
            Tuple of (timeseries data, labels)
        """
        data = []
        labels = []

        for i in range(n_samples):
            label = i % n_classes

            # Generate base time series with temporal correlations
            timeseries = self._generate_correlated_timeseries(n_timepoints, n_regions)

            # Add class-specific connectivity patterns
            if label == 1:
                # Altered connectivity for "patient" group
                timeseries = self._alter_connectivity(timeseries)

            data.append(timeseries)
            labels.append(label)

        return np.array(data), np.array(labels)

    def generate_multimodal_data(
        self,
        n_samples: int = 50,
        include_mri: bool = True,
        include_fmri: bool = True,
        include_clinical: bool = True,
        include_genetic: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate multi-modal neuroimaging dataset.

        Addresses Challenge #2: Heterogeneity of Data Sources
        and Challenge #8: Integration with Clinical and Genetic Data

        Args:
            n_samples: Number of samples
            include_*: Which modalities to include

        Returns:
            Dictionary with different modality data
        """
        data = {'labels': np.random.randint(0, 2, n_samples)}

        if include_mri:
            # Structural features (e.g., cortical thickness, volume)
            data['mri_features'] = np.random.randn(n_samples, 500)
            # Add class-related signal
            data['mri_features'][data['labels'] == 1, :50] += 0.5

        if include_fmri:
            # Functional connectivity matrices (flattened upper triangle)
            n_connections = 100 * 99 // 2
            data['fmri_connectivity'] = np.random.randn(n_samples, n_connections)
            # Add class-related signal
            data['fmri_connectivity'][data['labels'] == 1, :100] += 0.3

        if include_clinical:
            # Clinical scores (age, cognitive scores, etc.)
            data['clinical'] = np.random.randn(n_samples, 20)
            data['clinical'][:, 0] = np.random.uniform(20, 80, n_samples)  # Age
            data['clinical'][data['labels'] == 1, 1:5] -= 0.5  # Cognitive decline

        if include_genetic:
            # Genetic risk scores or SNP data
            data['genetic'] = np.random.binomial(2, 0.3, (n_samples, 100))
            # Add risk alleles for disease group
            data['genetic'][data['labels'] == 1, :10] = np.random.binomial(2, 0.6, (sum(data['labels'] == 1), 10))

        return data

    def generate_eeg_data(
        self,
        n_samples: int = 100,
        n_channels: int = 64,
        n_timepoints: int = 1000,
        sampling_rate: int = 256,
        n_classes: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EEG data with realistic frequency bands.

        Args:
            n_samples: Number of trials/subjects
            n_channels: Number of EEG channels
            n_timepoints: Number of time samples
            sampling_rate: Sampling rate in Hz
            n_classes: Number of classes

        Returns:
            Tuple of (EEG data, labels)
        """
        data = []
        labels = []

        for i in range(n_samples):
            label = i % n_classes

            # Generate signal with multiple frequency bands
            eeg = np.zeros((n_channels, n_timepoints))
            t = np.linspace(0, n_timepoints / sampling_rate, n_timepoints)

            for ch in range(n_channels):
                # Delta (0.5-4 Hz)
                eeg[ch] += 0.5 * np.sin(2 * np.pi * 2 * t + np.random.rand() * 2 * np.pi)
                # Theta (4-8 Hz)
                eeg[ch] += 0.3 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
                # Alpha (8-13 Hz)
                alpha_amp = 0.4 if label == 0 else 0.2  # Reduced alpha in "patients"
                eeg[ch] += alpha_amp * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
                # Beta (13-30 Hz)
                eeg[ch] += 0.2 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
                # Add noise
                eeg[ch] += 0.1 * np.random.randn(n_timepoints)

            data.append(eeg)
            labels.append(label)

        return np.array(data), np.array(labels)

    def _create_brain_structure(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Create a brain-like 3D structure."""
        volume = np.zeros(shape)
        center = np.array(shape) // 2

        # Create ellipsoid brain shape
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    dist = np.sqrt(
                        ((x - center[0]) / (shape[0] * 0.4)) ** 2 +
                        ((y - center[1]) / (shape[1] * 0.35)) ** 2 +
                        ((z - center[2]) / (shape[2] * 0.3)) ** 2
                    )
                    if dist < 1:
                        volume[x, y, z] = 1 - 0.5 * dist

        # Add internal structure variation
        volume = gaussian_filter(volume, sigma=2)
        volume += 0.1 * np.random.randn(*shape)

        return volume

    def _add_lesion_pattern(self, volume: np.ndarray) -> np.ndarray:
        """Add a lesion-like pattern to the volume."""
        shape = volume.shape

        # Random lesion location
        lesion_center = [
            np.random.randint(shape[0] // 4, 3 * shape[0] // 4),
            np.random.randint(shape[1] // 4, 3 * shape[1] // 4),
            np.random.randint(shape[2] // 4, 3 * shape[2] // 4)
        ]
        lesion_radius = np.random.randint(3, 8)

        # Create spherical lesion
        for x in range(max(0, lesion_center[0] - lesion_radius),
                       min(shape[0], lesion_center[0] + lesion_radius)):
            for y in range(max(0, lesion_center[1] - lesion_radius),
                          min(shape[1], lesion_center[1] + lesion_radius)):
                for z in range(max(0, lesion_center[2] - lesion_radius),
                              min(shape[2], lesion_center[2] + lesion_radius)):
                    dist = np.sqrt(
                        (x - lesion_center[0]) ** 2 +
                        (y - lesion_center[1]) ** 2 +
                        (z - lesion_center[2]) ** 2
                    )
                    if dist < lesion_radius:
                        volume[x, y, z] *= 0.5  # Reduced intensity

        return volume

    def _add_noise(self, volume: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to the volume."""
        noise = noise_level * np.random.randn(*volume.shape)
        return volume + noise

    def _generate_correlated_timeseries(
        self,
        n_timepoints: int,
        n_regions: int
    ) -> np.ndarray:
        """Generate temporally correlated time series."""
        # Create correlation structure
        cov = np.eye(n_regions)
        # Add some inter-region correlations
        for i in range(n_regions - 1):
            cov[i, i + 1] = 0.3
            cov[i + 1, i] = 0.3

        # Generate correlated data
        timeseries = np.random.multivariate_normal(
            np.zeros(n_regions), cov, n_timepoints
        )

        # Add temporal autocorrelation
        for t in range(1, n_timepoints):
            timeseries[t] = 0.3 * timeseries[t - 1] + 0.7 * timeseries[t]

        return timeseries

    def _alter_connectivity(self, timeseries: np.ndarray) -> np.ndarray:
        """Alter connectivity patterns for disease simulation."""
        # Increase variance in some regions
        timeseries[:, :20] *= 1.5
        # Decrease correlations between hemispheres (simulated)
        timeseries[:, 50:] += 0.2 * np.random.randn(*timeseries[:, 50:].shape)
        return timeseries
