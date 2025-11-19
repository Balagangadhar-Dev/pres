"""
Dimensionality Reduction Module

Addresses Challenge #1: High Dimensionality of Data
Implements various techniques to reduce neuroimaging data dimensions
while preserving relevant information.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import warnings


class DimensionalityReducer:
    """
    Comprehensive dimensionality reduction toolkit for neuroimaging data.

    Addresses the curse of dimensionality where features >> samples.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the dimensionality reducer.

        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self.fitted_models = {}

    def pca_reduction(
        self,
        data: np.ndarray,
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Principal Component Analysis for linear dimensionality reduction.

        Args:
            data: Input data (n_samples, n_features)
            n_components: Number of components (if None, use variance_threshold)
            variance_threshold: Minimum explained variance to retain

        Returns:
            Tuple of (transformed data, info dict with explained variance etc.)
        """
        if n_components is None:
            # First fit with all components to determine optimal number
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(data)

            # Find number of components for desired variance
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            n_components = max(1, min(n_components, min(data.shape)))

        pca = PCA(n_components=n_components, random_state=self.random_state)
        transformed = pca.fit_transform(data)

        self.fitted_models['pca'] = pca

        info = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_variance_explained': np.sum(pca.explained_variance_ratio_),
            'original_shape': data.shape,
            'reduced_shape': transformed.shape,
            'reduction_ratio': data.shape[1] / n_components
        }

        return transformed, info

    def ica_reduction(
        self,
        data: np.ndarray,
        n_components: int = 20,
        max_iter: int = 500
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Independent Component Analysis for source separation.

        Useful for separating brain signals from artifacts.

        Args:
            data: Input data (n_samples, n_features)
            n_components: Number of independent components
            max_iter: Maximum iterations for convergence

        Returns:
            Tuple of (independent components, mixing matrix info)
        """
        # Ensure n_components doesn't exceed data dimensions
        n_components = min(n_components, min(data.shape))

        ica = FastICA(
            n_components=n_components,
            random_state=self.random_state,
            max_iter=max_iter
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = ica.fit_transform(data)

        self.fitted_models['ica'] = ica

        info = {
            'n_components': n_components,
            'mixing_matrix_shape': ica.mixing_.shape,
            'n_iterations': ica.n_iter_,
            'original_shape': data.shape,
            'reduced_shape': transformed.shape
        }

        return transformed, info

    def tsne_visualization(
        self,
        data: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        t-SNE for visualization of high-dimensional data.

        Note: t-SNE is for visualization only, not for downstream ML tasks.

        Args:
            data: Input data (n_samples, n_features)
            n_components: Output dimensions (usually 2 or 3)
            perplexity: Balance between local and global structure
            n_iter: Number of iterations

        Returns:
            Tuple of (embedded data, info dict)
        """
        # Pre-reduce with PCA if data is very high-dimensional
        if data.shape[1] > 50:
            pca = PCA(n_components=50, random_state=self.random_state)
            data_reduced = pca.fit_transform(data)
        else:
            data_reduced = data

        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(data) - 1),
            n_iter=n_iter,
            random_state=self.random_state
        )

        embedded = tsne.fit_transform(data_reduced)

        info = {
            'n_components': n_components,
            'perplexity': perplexity,
            'kl_divergence': tsne.kl_divergence_,
            'original_shape': data.shape,
            'embedded_shape': embedded.shape
        }

        return embedded, info

    def feature_selection(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        k: int = 100,
        method: str = 'mutual_info'
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Select most informative features based on target variable.

        Args:
            data: Input features (n_samples, n_features)
            labels: Target labels (n_samples,)
            k: Number of features to select
            method: Selection method ('mutual_info')

        Returns:
            Tuple of (selected features, feature indices, info dict)
        """
        k = min(k, data.shape[1])

        if method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=k
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        selected_data = selector.fit_transform(data, labels)
        selected_indices = selector.get_support(indices=True)
        scores = selector.scores_

        self.fitted_models['feature_selector'] = selector

        info = {
            'n_selected': k,
            'selected_indices': selected_indices,
            'feature_scores': scores,
            'top_features': selected_indices[np.argsort(scores[selected_indices])[::-1]],
            'original_shape': data.shape,
            'selected_shape': selected_data.shape
        }

        return selected_data, selected_indices, info

    def autoencoder_reduction(
        self,
        data: np.ndarray,
        encoding_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Deep autoencoder for nonlinear dimensionality reduction.

        Uses a simple neural network architecture.

        Args:
            data: Input data (n_samples, n_features)
            encoding_dim: Dimension of the encoded representation
            epochs: Training epochs
            batch_size: Batch size for training

        Returns:
            Tuple of (encoded data, training info)
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch required for autoencoder. Install with: pip install torch")

        # Define autoencoder architecture
        input_dim = data.shape[1]

        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, encoding_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(encoding_dim, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, input_dim)
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

            def encode(self, x):
                return self.encoder(x)

        # Prepare data
        torch.manual_seed(self.random_state)
        tensor_data = torch.FloatTensor(data)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train
        model = Autoencoder()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_data, _ in dataloader:
                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(dataloader))

        # Get encoded representation
        model.eval()
        with torch.no_grad():
            encoded_data = model.encode(tensor_data).numpy()

        self.fitted_models['autoencoder'] = model

        info = {
            'encoding_dim': encoding_dim,
            'final_loss': losses[-1],
            'loss_history': losses,
            'original_shape': data.shape,
            'encoded_shape': encoded_data.shape,
            'compression_ratio': input_dim / encoding_dim
        }

        return encoded_data, info

    def compare_methods(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        target_dim: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different dimensionality reduction methods.

        Args:
            data: Input data
            labels: Target labels
            target_dim: Target number of dimensions

        Returns:
            Dictionary with results from each method
        """
        results = {}

        # PCA
        pca_data, pca_info = self.pca_reduction(data, n_components=target_dim)
        results['PCA'] = {
            'data': pca_data,
            'info': pca_info,
            'variance_explained': pca_info['total_variance_explained']
        }

        # ICA
        ica_data, ica_info = self.ica_reduction(data, n_components=target_dim)
        results['ICA'] = {
            'data': ica_data,
            'info': ica_info
        }

        # Feature Selection
        fs_data, fs_indices, fs_info = self.feature_selection(data, labels, k=target_dim)
        results['Feature_Selection'] = {
            'data': fs_data,
            'info': fs_info,
            'selected_features': fs_indices
        }

        return results

    def transform(self, data: np.ndarray, method: str = 'pca') -> np.ndarray:
        """
        Transform new data using a previously fitted model.

        Args:
            data: New data to transform
            method: Which fitted model to use

        Returns:
            Transformed data
        """
        if method not in self.fitted_models:
            raise ValueError(f"Model '{method}' not fitted. Available: {list(self.fitted_models.keys())}")

        model = self.fitted_models[method]

        if method == 'autoencoder':
            import torch
            with torch.no_grad():
                return model.encode(torch.FloatTensor(data)).numpy()
        elif method == 'feature_selector':
            return model.transform(data)
        else:
            return model.transform(data)
