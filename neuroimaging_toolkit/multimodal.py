"""
Multi-Modal Fusion Module for Neuroimaging

Addresses Challenge #2: Heterogeneity of Data Sources
and Challenge #8: Integration with Clinical and Genetic Data

Implements techniques for combining data from multiple imaging modalities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings


class MultiModalFusion:
    """
    Multi-modal data fusion techniques for neuroimaging.

    Combines data from different sources (MRI, fMRI, EEG, clinical, genetic)
    to improve predictive performance and understanding.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the multi-modal fusion system.

        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.fusion_model = None

    def early_fusion(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray,
        normalize: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Early fusion: Concatenate features from all modalities.

        Simple but effective when modalities have similar scales.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels
            normalize: Whether to normalize each modality

        Returns:
            Tuple of (fused features, fusion info)
        """
        fused_features = []
        modality_info = {}

        for name, data in modalities.items():
            if normalize:
                scaler = StandardScaler()
                data_normalized = scaler.fit_transform(data)
                self.scalers[name] = scaler
            else:
                data_normalized = data

            fused_features.append(data_normalized)
            modality_info[name] = {
                'n_features': data.shape[1],
                'start_idx': sum(m['n_features'] for m in modality_info.values()),
                'mean': np.mean(data),
                'std': np.std(data)
            }

        # Concatenate
        fused = np.hstack(fused_features)

        info = {
            'method': 'early_fusion',
            'n_modalities': len(modalities),
            'total_features': fused.shape[1],
            'modality_info': modality_info
        }

        return fused, info

    def late_fusion(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray,
        base_classifier: str = 'random_forest',
        fusion_method: str = 'average'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Late fusion: Train separate models, combine predictions.

        Good when modalities require different processing.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels
            base_classifier: Type of classifier for each modality
            fusion_method: How to combine predictions ('average', 'vote', 'stack')

        Returns:
            Tuple of (fused predictions, fusion info)
        """
        modality_predictions = {}
        modality_scores = {}

        for name, data in modalities.items():
            # Normalize
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)
            self.scalers[name] = scaler

            # Train classifier
            if base_classifier == 'random_forest':
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                )
            else:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(random_state=self.random_state, max_iter=1000)

            # Cross-validation score
            cv_scores = cross_val_score(clf, data_norm, labels, cv=5)
            modality_scores[name] = np.mean(cv_scores)

            # Fit and get predictions
            clf.fit(data_norm, labels)
            proba = clf.predict_proba(data_norm)
            modality_predictions[name] = proba

        # Fuse predictions
        if fusion_method == 'average':
            # Simple average
            fused_proba = np.mean(list(modality_predictions.values()), axis=0)
        elif fusion_method == 'weighted':
            # Weight by CV accuracy
            total_score = sum(modality_scores.values())
            fused_proba = sum(
                score / total_score * pred
                for name, (score, pred) in
                zip(modality_scores.keys(),
                    zip(modality_scores.values(), modality_predictions.values()))
            )
        elif fusion_method == 'vote':
            # Majority voting
            votes = np.stack([np.argmax(pred, axis=1) for pred in modality_predictions.values()])
            fused_proba = np.zeros((len(labels), len(np.unique(labels))))
            for i in range(len(labels)):
                unique, counts = np.unique(votes[:, i], return_counts=True)
                fused_proba[i, unique] = counts / len(modality_predictions)
        else:
            fused_proba = np.mean(list(modality_predictions.values()), axis=0)

        info = {
            'method': 'late_fusion',
            'fusion_method': fusion_method,
            'modality_scores': modality_scores,
            'n_modalities': len(modalities)
        }

        return fused_proba, info

    def intermediate_fusion(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray,
        n_components_per_modality: int = 20
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Intermediate fusion: Learn shared representation.

        Uses PCA to reduce each modality, then concatenates.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels
            n_components_per_modality: PCA components per modality

        Returns:
            Tuple of (fused representation, fusion info)
        """
        fused_components = []
        modality_variance = {}

        for name, data in modalities.items():
            # Normalize
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)
            self.scalers[name] = scaler

            # PCA
            n_comp = min(n_components_per_modality, min(data.shape))
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            components = pca.fit_transform(data_norm)

            fused_components.append(components)
            modality_variance[name] = np.sum(pca.explained_variance_ratio_)

        # Concatenate
        fused = np.hstack(fused_components)

        info = {
            'method': 'intermediate_fusion',
            'n_components_per_modality': n_components_per_modality,
            'total_components': fused.shape[1],
            'modality_variance_explained': modality_variance
        }

        return fused, info

    def attention_fusion(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Attention-based fusion: Learn modality importance.

        Learns to weight modalities based on their relevance.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels

        Returns:
            Tuple of (fused representation, attention weights info)
        """
        # First, reduce dimensionality of each modality to same size
        target_dim = 50
        reduced_modalities = []
        modality_names = list(modalities.keys())

        for name, data in modalities.items():
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)
            self.scalers[name] = scaler

            n_comp = min(target_dim, min(data.shape))
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            reduced = pca.fit_transform(data_norm)

            # Pad if needed
            if reduced.shape[1] < target_dim:
                padding = np.zeros((reduced.shape[0], target_dim - reduced.shape[1]))
                reduced = np.hstack([reduced, padding])

            reduced_modalities.append(reduced)

        # Stack modalities: (n_samples, n_modalities, n_features)
        stacked = np.stack(reduced_modalities, axis=1)

        # Learn attention weights using feature importance
        # Train classifier on each modality and use accuracy as attention
        attention_weights = []
        for i, name in enumerate(modality_names):
            clf = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            scores = cross_val_score(clf, reduced_modalities[i], labels, cv=3)
            attention_weights.append(np.mean(scores))

        # Normalize attention weights
        attention_weights = np.array(attention_weights)
        attention_weights = attention_weights / attention_weights.sum()

        # Apply attention
        fused = np.zeros((stacked.shape[0], stacked.shape[2]))
        for i, weight in enumerate(attention_weights):
            fused += weight * stacked[:, i, :]

        info = {
            'method': 'attention_fusion',
            'attention_weights': dict(zip(modality_names, attention_weights)),
            'target_dim': target_dim,
            'n_modalities': len(modalities)
        }

        return fused, info

    def tensor_fusion(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray,
        output_dim: int = 100
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Tensor fusion: Capture interactions between modalities.

        Creates outer product of modality representations.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels
            output_dim: Output dimension after fusion

        Returns:
            Tuple of (fused tensor representation, info)
        """
        # Reduce each modality first
        reduced_dim = 10
        reduced = []

        for name, data in modalities.items():
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)
            self.scalers[name] = scaler

            n_comp = min(reduced_dim, min(data.shape))
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            r = pca.fit_transform(data_norm)
            reduced.append(r)

        # For two modalities, compute outer product
        if len(reduced) == 2:
            n_samples = reduced[0].shape[0]
            outer = np.zeros((n_samples, reduced[0].shape[1] * reduced[1].shape[1]))
            for i in range(n_samples):
                outer[i] = np.outer(reduced[0][i], reduced[1][i]).flatten()

            # Reduce dimension of outer product
            if outer.shape[1] > output_dim:
                pca_out = PCA(n_components=output_dim, random_state=self.random_state)
                fused = pca_out.fit_transform(outer)
            else:
                fused = outer
        else:
            # For more modalities, use sequential outer products
            fused = reduced[0]
            for i in range(1, len(reduced)):
                n_samples = fused.shape[0]
                outer = np.zeros((n_samples, fused.shape[1] * reduced[i].shape[1]))
                for j in range(n_samples):
                    outer[j] = np.outer(fused[j], reduced[i][j]).flatten()

                # Reduce to manageable size
                if outer.shape[1] > output_dim:
                    pca_out = PCA(n_components=output_dim, random_state=self.random_state)
                    fused = pca_out.fit_transform(outer)
                else:
                    fused = outer

        info = {
            'method': 'tensor_fusion',
            'output_dim': fused.shape[1],
            'n_modalities': len(modalities)
        }

        return fused, info

    def hierarchical_fusion(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray,
        hierarchy: List[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Hierarchical fusion: Fuse in stages based on modality similarity.

        E.g., first fuse imaging modalities, then with clinical data.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels
            hierarchy: List of modality groups to fuse in order

        Returns:
            Tuple of (hierarchically fused data, info)
        """
        if hierarchy is None:
            # Default: all at once
            hierarchy = [list(modalities.keys())]

        fused_stages = []
        stage_info = []

        for stage_idx, stage_modalities in enumerate(hierarchy):
            stage_data = {name: modalities[name] for name in stage_modalities if name in modalities}

            if not stage_data:
                continue

            # Fuse this stage
            if len(stage_data) > 1:
                stage_fused, stage_fused_info = self.intermediate_fusion(
                    stage_data, labels, n_components_per_modality=30
                )
            else:
                # Single modality
                name = list(stage_data.keys())[0]
                data = stage_data[name]
                scaler = StandardScaler()
                stage_fused = scaler.fit_transform(data)
                stage_fused_info = {'method': 'single', 'modality': name}

            fused_stages.append(stage_fused)
            stage_info.append(stage_fused_info)

        # Final fusion of all stages
        if len(fused_stages) > 1:
            final_fused = np.hstack(fused_stages)
        else:
            final_fused = fused_stages[0]

        info = {
            'method': 'hierarchical_fusion',
            'n_stages': len(hierarchy),
            'stage_info': stage_info,
            'final_dim': final_fused.shape[1]
        }

        return final_fused, info

    def compare_fusion_methods(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare different fusion methods.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels

        Returns:
            Dictionary with results from each fusion method
        """
        results = {}

        # Test each fusion method
        methods = [
            ('early', self.early_fusion),
            ('intermediate', self.intermediate_fusion),
            ('attention', self.attention_fusion)
        ]

        for method_name, method_func in methods:
            try:
                fused, info = method_func(modalities, labels)

                # Evaluate with classifier
                clf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                )
                cv_scores = cross_val_score(clf, fused, labels, cv=5)

                results[method_name] = {
                    'accuracy': np.mean(cv_scores),
                    'std': np.std(cv_scores),
                    'n_features': fused.shape[1],
                    'info': info
                }
            except Exception as e:
                results[method_name] = {'error': str(e)}

        # Also test individual modalities
        for name, data in modalities.items():
            scaler = StandardScaler()
            data_norm = scaler.fit_transform(data)

            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
            cv_scores = cross_val_score(clf, data_norm, labels, cv=5)

            results[f'single_{name}'] = {
                'accuracy': np.mean(cv_scores),
                'std': np.std(cv_scores),
                'n_features': data.shape[1]
            }

        return results

    def generate_fusion_report(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> str:
        """
        Generate comprehensive fusion comparison report.

        Args:
            modalities: Dictionary of modality name -> data array
            labels: Target labels

        Returns:
            Formatted report string
        """
        comparison = self.compare_fusion_methods(modalities, labels)

        report = []
        report.append("=" * 60)
        report.append("MULTI-MODAL FUSION COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"\nNumber of modalities: {len(modalities)}")
        report.append(f"Total samples: {len(labels)}")

        report.append("\n" + "-" * 60)
        report.append("INDIVIDUAL MODALITIES")
        report.append("-" * 60)

        for name, data in modalities.items():
            if f'single_{name}' in comparison:
                res = comparison[f'single_{name}']
                report.append(f"\n{name}:")
                report.append(f"  Features: {res['n_features']}")
                report.append(f"  Accuracy: {res['accuracy']:.3f} (+/- {res['std']:.3f})")

        report.append("\n" + "-" * 60)
        report.append("FUSION METHODS")
        report.append("-" * 60)

        fusion_methods = ['early', 'intermediate', 'attention']
        for method in fusion_methods:
            if method in comparison:
                res = comparison[method]
                if 'error' in res:
                    report.append(f"\n{method.upper()}: Error - {res['error']}")
                else:
                    report.append(f"\n{method.upper()} FUSION:")
                    report.append(f"  Features: {res['n_features']}")
                    report.append(f"  Accuracy: {res['accuracy']:.3f} (+/- {res['std']:.3f})")

        # Find best method
        best_method = None
        best_acc = 0
        for method, res in comparison.items():
            if 'accuracy' in res and res['accuracy'] > best_acc:
                best_acc = res['accuracy']
                best_method = method

        report.append("\n" + "=" * 60)
        report.append("RECOMMENDATION")
        report.append("=" * 60)
        report.append(f"\nBest performing method: {best_method}")
        report.append(f"Accuracy: {best_acc:.3f}")

        return "\n".join(report)
