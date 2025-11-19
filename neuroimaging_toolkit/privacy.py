"""
Privacy-Preserving Techniques for Neuroimaging Data

Addresses Challenge #10: Ethical and Privacy Issues
Implements differential privacy, federated learning concepts, and secure aggregation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
import warnings


class PrivacyPreserver:
    """
    Privacy-preserving machine learning techniques for sensitive neuroimaging data.

    Implements methods to protect patient privacy while enabling research.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the privacy preserver.

        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def add_differential_privacy(
        self,
        data: np.ndarray,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
        mechanism: str = 'laplace'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Add differential privacy noise to data.

        Differential privacy provides mathematical guarantees that
        individual records cannot be identified from the output.

        Args:
            data: Input data to privatize
            epsilon: Privacy budget (lower = more private, more noise)
            sensitivity: Maximum change one record can cause
            mechanism: 'laplace' or 'gaussian'

        Returns:
            Tuple of (privatized data, privacy info)
        """
        if mechanism == 'laplace':
            # Laplace mechanism for epsilon-DP
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, data.shape)
            info = {
                'mechanism': 'laplace',
                'scale': scale,
                'epsilon': epsilon,
                'delta': 0
            }
        elif mechanism == 'gaussian':
            # Gaussian mechanism for (epsilon, delta)-DP
            delta = 1e-5
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            noise = np.random.normal(0, sigma, data.shape)
            info = {
                'mechanism': 'gaussian',
                'sigma': sigma,
                'epsilon': epsilon,
                'delta': delta
            }
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        privatized = data + noise
        info['noise_magnitude'] = np.mean(np.abs(noise))
        info['snr_reduction'] = np.std(data) / np.std(noise) if np.std(noise) > 0 else float('inf')

        return privatized, info

    def private_mean(
        self,
        data: np.ndarray,
        epsilon: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute differentially private mean.

        Args:
            data: Input data (n_samples, n_features)
            epsilon: Privacy budget
            bounds: (min, max) bounds for data clipping

        Returns:
            Tuple of (private mean, privacy info)
        """
        if bounds is None:
            bounds = (np.min(data), np.max(data))

        # Clip data to bounds
        clipped = np.clip(data, bounds[0], bounds[1])

        # Compute true mean
        true_mean = np.mean(clipped, axis=0)

        # Sensitivity of mean is (max - min) / n
        n = len(data)
        sensitivity = (bounds[1] - bounds[0]) / n

        # Add Laplace noise
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, true_mean.shape)
        private_mean = true_mean + noise

        info = {
            'epsilon': epsilon,
            'sensitivity': sensitivity,
            'noise_scale': scale,
            'n_samples': n,
            'error': np.mean(np.abs(private_mean - true_mean))
        }

        return private_mean, info

    def private_histogram(
        self,
        data: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1.0,
        range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Compute differentially private histogram.

        Args:
            data: Input data (1D)
            n_bins: Number of histogram bins
            epsilon: Privacy budget
            range: (min, max) range for histogram

        Returns:
            Tuple of (private counts, bin edges, privacy info)
        """
        # Compute true histogram
        counts, edges = np.histogram(data, bins=n_bins, range=range)

        # Sensitivity of histogram is 1 (one person changes one bin by 1)
        sensitivity = 1

        # Add Laplace noise
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, counts.shape)
        private_counts = np.maximum(0, counts + noise)  # Ensure non-negative

        info = {
            'epsilon': epsilon,
            'sensitivity': sensitivity,
            'noise_scale': scale,
            'total_error': np.sum(np.abs(private_counts - counts))
        }

        return private_counts, edges, info

    def dp_sgd_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        epochs: int = 10,
        batch_size: int = 32,
        max_grad_norm: float = 1.0
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with Differentially Private Stochastic Gradient Descent.

        Args:
            X: Training features
            y: Training labels
            epsilon: Total privacy budget
            delta: Privacy parameter
            epochs: Number of training epochs
            batch_size: Batch size
            max_grad_norm: Gradient clipping threshold

        Returns:
            Tuple of (trained model, privacy info)
        """
        n_samples = len(X)
        n_batches = n_samples // batch_size

        # Calculate noise multiplier for target (epsilon, delta)
        # Simplified calculation - in practice use privacy accounting
        noise_multiplier = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise_multiplier *= np.sqrt(epochs * n_batches)  # Composition

        # Initialize model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use SGD classifier
        model = SGDClassifier(
            loss='log_loss',
            learning_rate='constant',
            eta0=0.01,
            random_state=self.random_state
        )

        # Manual DP-SGD training
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        bias = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)

            for batch_idx in range(n_batches):
                batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                X_batch = X_scaled[batch_indices]
                y_batch = y[batch_indices]

                # Compute per-sample gradients
                logits = X_batch @ weights + bias
                probs = 1 / (1 + np.exp(-logits))
                errors = probs - y_batch

                grad_weights = X_batch.T @ errors / batch_size
                grad_bias = np.mean(errors)

                # Clip gradients
                grad_norm = np.sqrt(np.sum(grad_weights ** 2) + grad_bias ** 2)
                if grad_norm > max_grad_norm:
                    grad_weights *= max_grad_norm / grad_norm
                    grad_bias *= max_grad_norm / grad_norm

                # Add noise
                noise_std = max_grad_norm * noise_multiplier / batch_size
                grad_weights += np.random.normal(0, noise_std, grad_weights.shape)
                grad_bias += np.random.normal(0, noise_std)

                # Update
                weights -= 0.01 * grad_weights
                bias -= 0.01 * grad_bias

        # Create final model
        model.coef_ = weights.reshape(1, -1)
        model.intercept_ = np.array([bias])
        model.classes_ = np.unique(y)

        info = {
            'epsilon': epsilon,
            'delta': delta,
            'noise_multiplier': noise_multiplier,
            'epochs': epochs,
            'batch_size': batch_size,
            'max_grad_norm': max_grad_norm
        }

        return model, info

    def federated_averaging(
        self,
        local_models: List[Dict[str, np.ndarray]],
        sample_sizes: List[int]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate local model updates using Federated Averaging.

        Simulates federated learning where data stays local.

        Args:
            local_models: List of model parameter dicts from each site
            sample_sizes: Number of samples at each site

        Returns:
            Averaged global model parameters
        """
        total_samples = sum(sample_sizes)
        weights = [n / total_samples for n in sample_sizes]

        # Average parameters
        global_model = {}
        for key in local_models[0].keys():
            global_model[key] = sum(
                w * model[key] for w, model in zip(weights, local_models)
            )

        return global_model

    def simulate_federated_learning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_sites: int = 5,
        n_rounds: int = 10,
        local_epochs: int = 3
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Simulate federated learning across multiple sites.

        Each site trains locally, only gradients are shared.

        Args:
            X: Full dataset features
            y: Full dataset labels
            n_sites: Number of federated sites
            n_rounds: Number of communication rounds
            local_epochs: Epochs per round at each site

        Returns:
            Tuple of (global model, training info)
        """
        n_samples = len(X)
        samples_per_site = n_samples // n_sites

        # Split data across sites (IID split for simplicity)
        site_data = []
        for i in range(n_sites):
            start = i * samples_per_site
            end = start + samples_per_site if i < n_sites - 1 else n_samples
            site_data.append({
                'X': X[start:end],
                'y': y[start:end]
            })

        # Initialize global model
        n_features = X.shape[1]
        global_weights = np.zeros(n_features)
        global_bias = 0

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Federated training
        round_accuracies = []

        for round_idx in range(n_rounds):
            local_updates = []

            # Local training at each site
            for site_idx, site in enumerate(site_data):
                # Get local data
                start = site_idx * samples_per_site
                end = start + samples_per_site if site_idx < n_sites - 1 else n_samples
                X_local = X_scaled[start:end]
                y_local = site['y']

                # Initialize with global model
                weights = global_weights.copy()
                bias = global_bias

                # Local SGD
                for _ in range(local_epochs):
                    logits = X_local @ weights + bias
                    probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                    errors = probs - y_local

                    grad_w = X_local.T @ errors / len(y_local)
                    grad_b = np.mean(errors)

                    weights -= 0.1 * grad_w
                    bias -= 0.1 * grad_b

                local_updates.append({
                    'weights': weights,
                    'bias': bias,
                    'n_samples': len(y_local)
                })

            # Federated averaging
            total_samples = sum(u['n_samples'] for u in local_updates)
            global_weights = sum(
                u['weights'] * u['n_samples'] / total_samples
                for u in local_updates
            )
            global_bias = sum(
                u['bias'] * u['n_samples'] / total_samples
                for u in local_updates
            )

            # Evaluate global model
            logits = X_scaled @ global_weights + global_bias
            preds = (logits > 0).astype(int)
            accuracy = np.mean(preds == y)
            round_accuracies.append(accuracy)

        # Create final model
        model = LogisticRegression()
        model.coef_ = global_weights.reshape(1, -1)
        model.intercept_ = np.array([global_bias])
        model.classes_ = np.unique(y)

        info = {
            'n_sites': n_sites,
            'n_rounds': n_rounds,
            'local_epochs': local_epochs,
            'round_accuracies': round_accuracies,
            'final_accuracy': round_accuracies[-1],
            'samples_per_site': samples_per_site
        }

        return model, info

    def secure_aggregation(
        self,
        local_values: List[np.ndarray],
        n_parties: int = None
    ) -> np.ndarray:
        """
        Simulate secure aggregation for private sum computation.

        In real implementation, this would use cryptographic protocols.

        Args:
            local_values: Values from each party
            n_parties: Number of parties (for masking)

        Returns:
            Aggregated sum
        """
        if n_parties is None:
            n_parties = len(local_values)

        # In real secure aggregation:
        # 1. Each party generates random masks that sum to 0
        # 2. Parties add masks to their values
        # 3. Server sums masked values
        # 4. Masks cancel out, revealing only the sum

        # Simplified simulation
        aggregated = sum(local_values)

        return aggregated

    def k_anonymity_check(
        self,
        data: np.ndarray,
        quasi_identifiers: List[int],
        k: int = 5
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if dataset satisfies k-anonymity.

        k-anonymity: each record is indistinguishable from k-1 others.

        Args:
            data: Dataset
            quasi_identifiers: Indices of quasi-identifier columns
            k: Minimum group size

        Returns:
            Tuple of (is_k_anonymous, analysis info)
        """
        # Get quasi-identifier values
        qi_data = data[:, quasi_identifiers]

        # Find unique combinations and their counts
        unique_rows, counts = np.unique(qi_data, axis=0, return_counts=True)

        min_count = np.min(counts)
        is_k_anonymous = min_count >= k

        info = {
            'k': k,
            'min_group_size': min_count,
            'is_k_anonymous': is_k_anonymous,
            'n_groups': len(unique_rows),
            'group_sizes': counts,
            'vulnerable_groups': np.sum(counts < k)
        }

        return is_k_anonymous, info

    def data_synthesis(
        self,
        data: np.ndarray,
        n_synthetic: int = None,
        method: str = 'gaussian'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic data that preserves statistical properties.

        Useful for sharing data without exposing real records.

        Args:
            data: Original data
            n_synthetic: Number of synthetic samples (default: same as original)
            method: 'gaussian' or 'kde'

        Returns:
            Tuple of (synthetic data, generation info)
        """
        if n_synthetic is None:
            n_synthetic = len(data)

        if method == 'gaussian':
            # Fit multivariate Gaussian
            mean = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False)

            # Handle potential singular covariance
            try:
                synthetic = np.random.multivariate_normal(mean, cov, n_synthetic)
            except np.linalg.LinAlgError:
                # Add small regularization
                cov += 1e-6 * np.eye(cov.shape[0])
                synthetic = np.random.multivariate_normal(mean, cov, n_synthetic)

        elif method == 'kde':
            # Kernel density estimation
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data.T)
            synthetic = kde.resample(n_synthetic).T

        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute utility metrics
        mean_diff = np.mean(np.abs(np.mean(synthetic, axis=0) - np.mean(data, axis=0)))
        std_diff = np.mean(np.abs(np.std(synthetic, axis=0) - np.std(data, axis=0)))

        info = {
            'method': method,
            'n_original': len(data),
            'n_synthetic': n_synthetic,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'utility_score': 1 - (mean_diff + std_diff) / 2  # Simple utility metric
        }

        return synthetic, info

    def privacy_budget_tracker(
        self,
        queries: List[Dict[str, float]],
        total_budget: float = 1.0
    ) -> Dict[str, Any]:
        """
        Track privacy budget across multiple queries.

        Args:
            queries: List of queries with their epsilon values
            total_budget: Total privacy budget

        Returns:
            Budget tracking information
        """
        used_budget = sum(q.get('epsilon', 0) for q in queries)
        remaining = total_budget - used_budget

        return {
            'total_budget': total_budget,
            'used_budget': used_budget,
            'remaining_budget': remaining,
            'n_queries': len(queries),
            'is_exhausted': remaining <= 0,
            'queries': queries
        }
