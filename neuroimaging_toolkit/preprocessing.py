"""
Preprocessing Module for Neuroimaging Data

Addresses Challenge #3: Noise and Artifacts
Implements detection and removal of common artifacts in brain imaging data.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter
from scipy.stats import zscore


class NoiseDetector:
    """
    Detect various types of noise and artifacts in neuroimaging data.
    """

    def __init__(self):
        """Initialize the noise detector."""
        pass

    def detect_motion_artifacts(
        self,
        timeseries: np.ndarray,
        threshold: float = 3.0
    ) -> Dict[str, any]:
        """
        Detect motion artifacts in fMRI time series.

        Motion causes sudden signal changes and framewise displacement.

        Args:
            timeseries: Time series data (n_timepoints, n_features) or (n_timepoints,)
            threshold: Z-score threshold for artifact detection

        Returns:
            Dictionary with artifact information
        """
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        # Calculate framewise displacement (derivative)
        fd = np.abs(np.diff(timeseries, axis=0))

        # Mean framewise displacement across features
        mean_fd = np.mean(fd, axis=1)

        # Z-score to find outliers
        z_scores = zscore(mean_fd)
        artifact_frames = np.where(np.abs(z_scores) > threshold)[0] + 1  # +1 for diff offset

        # Calculate DVARS (temporal derivative of timecourses)
        dvars = np.sqrt(np.mean(np.diff(timeseries, axis=0) ** 2, axis=1))

        return {
            'artifact_frames': artifact_frames,
            'n_artifacts': len(artifact_frames),
            'percentage_affected': 100 * len(artifact_frames) / len(timeseries),
            'framewise_displacement': mean_fd,
            'dvars': dvars,
            'mean_fd': np.mean(mean_fd),
            'max_fd': np.max(mean_fd)
        }

    def detect_signal_dropout(
        self,
        volume: np.ndarray,
        threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Detect signal dropout regions in 3D volumes.

        Common near air-tissue interfaces (sinuses, ear canals).

        Args:
            volume: 3D volume data
            threshold: Threshold below which signal is considered dropped

        Returns:
            Dictionary with dropout information
        """
        # Normalize volume
        volume_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Find low signal regions
        dropout_mask = volume_norm < threshold

        # Find connected components (simplified)
        dropout_voxels = np.sum(dropout_mask)
        total_voxels = volume.size

        return {
            'dropout_mask': dropout_mask,
            'dropout_voxels': dropout_voxels,
            'percentage_dropout': 100 * dropout_voxels / total_voxels,
            'dropout_locations': np.argwhere(dropout_mask)
        }

    def detect_spike_noise(
        self,
        timeseries: np.ndarray,
        threshold: float = 4.0
    ) -> Dict[str, any]:
        """
        Detect spike artifacts (sudden large amplitude changes).

        Args:
            timeseries: Time series data
            threshold: Z-score threshold for spike detection

        Returns:
            Dictionary with spike information
        """
        # Z-score the data
        z_data = zscore(timeseries, axis=0)

        # Find spikes
        if z_data.ndim == 1:
            spike_mask = np.abs(z_data) > threshold
        else:
            spike_mask = np.any(np.abs(z_data) > threshold, axis=1)

        spike_indices = np.where(spike_mask)[0]

        return {
            'spike_indices': spike_indices,
            'n_spikes': len(spike_indices),
            'percentage_spikes': 100 * len(spike_indices) / len(timeseries),
            'spike_amplitudes': np.max(np.abs(z_data), axis=-1) if z_data.ndim > 1 else np.abs(z_data)
        }

    def estimate_snr(
        self,
        data: np.ndarray,
        signal_region: Optional[np.ndarray] = None,
        noise_region: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate Signal-to-Noise Ratio.

        Args:
            data: Image or volume data
            signal_region: Boolean mask for signal region
            noise_region: Boolean mask for noise region

        Returns:
            Estimated SNR value
        """
        if signal_region is None:
            # Use center as signal region
            center_slice = tuple(slice(s//4, 3*s//4) for s in data.shape)
            signal_region = np.zeros(data.shape, dtype=bool)
            signal_region[center_slice] = True

        if noise_region is None:
            # Use corners as noise region
            noise_region = ~signal_region

        signal_mean = np.mean(data[signal_region])
        noise_std = np.std(data[noise_region])

        snr = signal_mean / (noise_std + 1e-8)

        return snr

    def analyze_frequency_content(
        self,
        timeseries: np.ndarray,
        sampling_rate: float = 1.0
    ) -> Dict[str, any]:
        """
        Analyze frequency content to identify noise sources.

        Args:
            timeseries: Time series data (1D)
            sampling_rate: Sampling rate in Hz

        Returns:
            Dictionary with frequency analysis results
        """
        # Compute power spectral density
        freqs, psd = signal.welch(timeseries, fs=sampling_rate, nperseg=min(256, len(timeseries)))

        # Find dominant frequencies
        peak_indices = signal.find_peaks(psd, height=np.mean(psd))[0]
        peak_freqs = freqs[peak_indices]
        peak_powers = psd[peak_indices]

        # Check for common artifact frequencies
        artifacts = []
        for freq in peak_freqs:
            if 0.1 <= freq <= 0.5:
                artifacts.append(f"Respiratory artifact at {freq:.2f} Hz")
            elif 0.8 <= freq <= 1.5:
                artifacts.append(f"Cardiac artifact at {freq:.2f} Hz")
            elif freq in [50, 60]:
                artifacts.append(f"Power line interference at {freq} Hz")

        return {
            'frequencies': freqs,
            'power_spectrum': psd,
            'peak_frequencies': peak_freqs,
            'peak_powers': peak_powers,
            'potential_artifacts': artifacts
        }


class ArtifactRemover:
    """
    Remove various artifacts from neuroimaging data.
    """

    def __init__(self):
        """Initialize the artifact remover."""
        pass

    def temporal_filtering(
        self,
        timeseries: np.ndarray,
        low_freq: float = 0.01,
        high_freq: float = 0.1,
        sampling_rate: float = 1.0
    ) -> np.ndarray:
        """
        Apply bandpass filtering to remove noise outside frequency range.

        Args:
            timeseries: Time series data (n_timepoints,) or (n_timepoints, n_features)
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            sampling_rate: Sampling rate (Hz)

        Returns:
            Filtered time series
        """
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Ensure valid frequency range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))

        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')

        # Apply filter
        if timeseries.ndim == 1:
            filtered = signal.filtfilt(b, a, timeseries)
        else:
            filtered = np.apply_along_axis(
                lambda x: signal.filtfilt(b, a, x), 0, timeseries
            )

        return filtered

    def spatial_smoothing(
        self,
        volume: np.ndarray,
        fwhm: float = 6.0
    ) -> np.ndarray:
        """
        Apply Gaussian spatial smoothing.

        Args:
            volume: 3D volume data
            fwhm: Full Width at Half Maximum in voxels

        Returns:
            Smoothed volume
        """
        # Convert FWHM to sigma
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        smoothed = gaussian_filter(volume, sigma=sigma)

        return smoothed

    def motion_regression(
        self,
        timeseries: np.ndarray,
        motion_params: np.ndarray
    ) -> np.ndarray:
        """
        Remove motion-related variance using regression.

        Args:
            timeseries: Time series data (n_timepoints, n_features)
            motion_params: Motion parameters (n_timepoints, n_params)

        Returns:
            Motion-corrected time series
        """
        # Add intercept
        design_matrix = np.column_stack([np.ones(len(motion_params)), motion_params])

        # Fit and remove motion effects
        if timeseries.ndim == 1:
            timeseries = timeseries.reshape(-1, 1)

        # Compute residuals
        beta = np.linalg.lstsq(design_matrix, timeseries, rcond=None)[0]
        predicted = design_matrix @ beta
        residuals = timeseries - predicted

        # Add back the mean
        residuals += np.mean(timeseries, axis=0)

        return residuals.squeeze()

    def spike_removal(
        self,
        timeseries: np.ndarray,
        threshold: float = 4.0,
        method: str = 'interpolate'
    ) -> np.ndarray:
        """
        Remove spike artifacts from time series.

        Args:
            timeseries: Time series data
            threshold: Z-score threshold for spike detection
            method: 'interpolate' or 'median'

        Returns:
            Despike time series
        """
        cleaned = timeseries.copy()

        # Find spikes
        z_data = zscore(timeseries, axis=0)
        if z_data.ndim == 1:
            spike_mask = np.abs(z_data) > threshold
        else:
            spike_mask = np.abs(z_data) > threshold

        # Replace spikes
        if method == 'interpolate':
            # Linear interpolation
            if timeseries.ndim == 1:
                spike_indices = np.where(spike_mask)[0]
                good_indices = np.where(~spike_mask)[0]
                if len(good_indices) > 0 and len(spike_indices) > 0:
                    cleaned[spike_indices] = np.interp(
                        spike_indices, good_indices, timeseries[good_indices]
                    )
            else:
                for i in range(timeseries.shape[1]):
                    spike_indices = np.where(spike_mask[:, i])[0]
                    good_indices = np.where(~spike_mask[:, i])[0]
                    if len(good_indices) > 0 and len(spike_indices) > 0:
                        cleaned[spike_indices, i] = np.interp(
                            spike_indices, good_indices, timeseries[good_indices, i]
                        )

        elif method == 'median':
            # Replace with local median
            if timeseries.ndim == 1:
                cleaned = median_filter(timeseries, size=5)
            else:
                for i in range(timeseries.shape[1]):
                    cleaned[:, i] = median_filter(timeseries[:, i], size=5)

        return cleaned

    def global_signal_regression(
        self,
        timeseries: np.ndarray
    ) -> np.ndarray:
        """
        Remove global signal from time series.

        Note: Controversial - can introduce spurious anticorrelations.

        Args:
            timeseries: Time series data (n_timepoints, n_features)

        Returns:
            GSR-corrected time series
        """
        if timeseries.ndim == 1:
            return timeseries

        # Compute global signal
        global_signal = np.mean(timeseries, axis=1, keepdims=True)

        # Regress out
        design = np.column_stack([np.ones(len(global_signal)), global_signal])
        beta = np.linalg.lstsq(design, timeseries, rcond=None)[0]
        predicted = design @ beta
        residuals = timeseries - predicted

        # Add back mean
        residuals += np.mean(timeseries, axis=0)

        return residuals

    def denoise_components(
        self,
        data: np.ndarray,
        n_components: int = 5,
        variance_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Remove noise components using PCA-based denoising.

        Removes components that explain little variance (likely noise).

        Args:
            data: Input data (n_samples, n_features)
            n_components: Number of components to keep
            variance_threshold: Minimum variance for component to be signal

        Returns:
            Denoised data
        """
        from sklearn.decomposition import PCA

        # Fit PCA
        pca = PCA(n_components=min(n_components, min(data.shape)))
        transformed = pca.fit_transform(data)

        # Keep only components above variance threshold
        keep_mask = pca.explained_variance_ratio_ > variance_threshold
        if not np.any(keep_mask):
            keep_mask[0] = True  # Keep at least one component

        # Reconstruct without noise components
        transformed[:, ~keep_mask] = 0
        denoised = pca.inverse_transform(transformed)

        return denoised

    def preprocess_pipeline(
        self,
        timeseries: np.ndarray,
        sampling_rate: float = 1.0,
        apply_bandpass: bool = True,
        apply_despiking: bool = True,
        apply_denoising: bool = True
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Apply complete preprocessing pipeline.

        Args:
            timeseries: Input time series
            sampling_rate: Sampling rate in Hz
            apply_bandpass: Whether to apply bandpass filtering
            apply_despiking: Whether to remove spikes
            apply_denoising: Whether to apply PCA denoising

        Returns:
            Tuple of (preprocessed data, processing info)
        """
        processed = timeseries.copy()
        steps_applied = []

        # Despiking
        if apply_despiking:
            processed = self.spike_removal(processed)
            steps_applied.append('despiking')

        # Bandpass filtering
        if apply_bandpass:
            processed = self.temporal_filtering(
                processed, sampling_rate=sampling_rate
            )
            steps_applied.append('bandpass_filtering')

        # PCA denoising
        if apply_denoising and processed.ndim > 1:
            processed = self.denoise_components(processed)
            steps_applied.append('pca_denoising')

        info = {
            'steps_applied': steps_applied,
            'input_shape': timeseries.shape,
            'output_shape': processed.shape
        }

        return processed, info
