"""
Interpretable AI Module for Neuroimaging

Addresses Challenge #5: Interpretability of Models
Implements explainable machine learning techniques for clinical applications.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import warnings


class InterpretableClassifier:
    """
    Collection of interpretable machine learning models for neuroimaging.

    Focuses on models that provide clear explanations for clinical use.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the interpretable classifier.

        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.feature_importance = {}

    def train_logistic_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty: str = 'l1',
        C: float = 1.0,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train L1-regularized logistic regression for sparse, interpretable solutions.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            penalty: Regularization type ('l1' for sparsity)
            C: Inverse regularization strength
            feature_names: Optional names for features

        Returns:
            Dictionary with model info and interpretations
        """
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver='saga',
            max_iter=1000,
            random_state=self.random_state
        )

        model.fit(X, y)
        self.models['logistic_regression'] = model

        # Get feature importance from coefficients
        coef = model.coef_.flatten()
        importance = np.abs(coef)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]

        # Calculate odds ratios for interpretation
        odds_ratios = np.exp(coef)

        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=5)

        self.feature_importance['logistic_regression'] = importance

        return {
            'model': model,
            'coefficients': coef,
            'feature_importance': importance,
            'top_features': [(feature_names[i], importance[i], coef[i]) for i in sorted_idx[:20]],
            'odds_ratios': odds_ratios,
            'n_nonzero_features': np.sum(coef != 0),
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'interpretation': self._interpret_logistic(coef, feature_names, sorted_idx[:5])
        }

    def train_decision_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_depth: int = 5,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train decision tree for rule-based interpretation.

        Args:
            X: Training features
            y: Training labels
            max_depth: Maximum tree depth (controls complexity)
            feature_names: Optional names for features

        Returns:
            Dictionary with model info and decision rules
        """
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state
        )

        model.fit(X, y)
        self.models['decision_tree'] = model

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Get feature importance
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        # Extract decision rules
        rules = self._extract_tree_rules(model, feature_names)

        cv_scores = cross_val_score(model, X, y, cv=5)

        self.feature_importance['decision_tree'] = importance

        return {
            'model': model,
            'feature_importance': importance,
            'top_features': [(feature_names[i], importance[i]) for i in sorted_idx[:20]],
            'decision_rules': rules,
            'tree_depth': model.get_depth(),
            'n_leaves': model.get_n_leaves(),
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }

    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 10,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train random forest with feature importance analysis.

        Args:
            X: Training features
            y: Training labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            feature_names: Optional names for features

        Returns:
            Dictionary with model info and feature importance
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X, y)
        self.models['random_forest'] = model

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Gini importance
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]

        # Permutation importance (more reliable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=self.random_state, n_jobs=-1
            )

        cv_scores = cross_val_score(model, X, y, cv=5)

        self.feature_importance['random_forest'] = importance

        return {
            'model': model,
            'gini_importance': importance,
            'permutation_importance': perm_importance.importances_mean,
            'top_features_gini': [(feature_names[i], importance[i]) for i in sorted_idx[:20]],
            'top_features_permutation': [
                (feature_names[i], perm_importance.importances_mean[i])
                for i in np.argsort(perm_importance.importances_mean)[::-1][:20]
            ],
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores)
        }

    def compute_shap_values(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for model interpretation.

        SHAP provides consistent, locally accurate feature attributions.

        Args:
            model_name: Name of fitted model
            X: Data to explain
            feature_names: Optional feature names

        Returns:
            Dictionary with SHAP analysis results
        """
        try:
            import shap
        except ImportError:
            # Provide approximate importance instead
            return self._approximate_shap(model_name, X, feature_names)

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Create explainer based on model type
        if model_name in ['random_forest', 'gradient_boosting']:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                shap.sample(X, 100)
            )

        shap_values = explainer.shap_values(X)

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        # Mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_shap)[::-1]

        return {
            'shap_values': shap_values,
            'mean_abs_shap': mean_shap,
            'top_features': [(feature_names[i], mean_shap[i]) for i in sorted_idx[:20]],
            'feature_names': feature_names
        }

    def _approximate_shap(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Approximate SHAP-like importance when SHAP library not available."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Use permutation importance as approximation
        y_pred = model.predict(X)
        perm_imp = permutation_importance(
            model, X, y_pred, n_repeats=10, random_state=self.random_state
        )

        sorted_idx = np.argsort(perm_imp.importances_mean)[::-1]

        return {
            'approximate_importance': perm_imp.importances_mean,
            'top_features': [
                (feature_names[i], perm_imp.importances_mean[i])
                for i in sorted_idx[:20]
            ],
            'note': 'SHAP library not available, using permutation importance'
        }

    def attention_visualization(
        self,
        model,
        X: np.ndarray,
        layer_name: str = 'attention'
    ) -> np.ndarray:
        """
        Extract attention weights from neural network models.

        For use with attention-based deep learning models.

        Args:
            model: PyTorch model with attention layers
            X: Input data
            layer_name: Name of attention layer

        Returns:
            Attention weights
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for attention visualization")

        model.eval()
        attention_weights = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                attention_weights.append(output[1].detach().cpu().numpy())
            else:
                attention_weights.append(output.detach().cpu().numpy())

        # Register hook
        for name, module in model.named_modules():
            if layer_name in name:
                module.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = model(torch.FloatTensor(X))

        return np.array(attention_weights) if attention_weights else None

    def generate_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate comprehensive interpretability report.

        Args:
            X: Feature data
            y: Labels
            feature_names: Feature names

        Returns:
            Formatted report string
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        report = []
        report.append("=" * 60)
        report.append("NEUROIMAGING CLASSIFICATION INTERPRETABILITY REPORT")
        report.append("=" * 60)
        report.append(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
        report.append(f"Classes: {np.unique(y)}")
        report.append(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Train and report on each model
        report.append("\n" + "-" * 60)
        report.append("LOGISTIC REGRESSION (Sparse, Linear)")
        report.append("-" * 60)
        lr_results = self.train_logistic_regression(X, y, feature_names=feature_names)
        report.append(f"CV Accuracy: {lr_results['cv_accuracy']:.3f} (+/- {lr_results['cv_std']:.3f})")
        report.append(f"Non-zero features: {lr_results['n_nonzero_features']}/{X.shape[1]}")
        report.append("\nTop 10 predictive features:")
        for name, imp, coef in lr_results['top_features'][:10]:
            direction = "+" if coef > 0 else "-"
            report.append(f"  {direction} {name}: {imp:.4f}")

        report.append("\n" + "-" * 60)
        report.append("DECISION TREE (Rule-based)")
        report.append("-" * 60)
        dt_results = self.train_decision_tree(X, y, feature_names=feature_names)
        report.append(f"CV Accuracy: {dt_results['cv_accuracy']:.3f} (+/- {dt_results['cv_std']:.3f})")
        report.append(f"Tree depth: {dt_results['tree_depth']}, Leaves: {dt_results['n_leaves']}")
        report.append("\nDecision rules:")
        for rule in dt_results['decision_rules'][:5]:
            report.append(f"  {rule}")

        report.append("\n" + "-" * 60)
        report.append("RANDOM FOREST (Ensemble)")
        report.append("-" * 60)
        rf_results = self.train_random_forest(X, y, feature_names=feature_names)
        report.append(f"CV Accuracy: {rf_results['cv_accuracy']:.3f} (+/- {rf_results['cv_std']:.3f})")
        report.append("\nTop 10 features (Gini importance):")
        for name, imp in rf_results['top_features_gini'][:10]:
            report.append(f"  {name}: {imp:.4f}")

        report.append("\n" + "=" * 60)
        report.append("CLINICAL INTERPRETATION SUMMARY")
        report.append("=" * 60)

        # Find consensus important features
        all_importances = {}
        for name in feature_names:
            all_importances[name] = []

        for i, name in enumerate(feature_names):
            all_importances[name].append(lr_results['feature_importance'][i])
            all_importances[name].append(dt_results['feature_importance'][i])
            all_importances[name].append(rf_results['gini_importance'][i])

        # Average importance across models
        avg_importance = {
            name: np.mean(imps) for name, imps in all_importances.items()
        }
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

        report.append("\nConsensus top features (averaged across models):")
        for name, imp in sorted_features[:10]:
            report.append(f"  {name}: {imp:.4f}")

        return "\n".join(report)

    def _interpret_logistic(
        self,
        coef: np.ndarray,
        feature_names: List[str],
        top_indices: np.ndarray
    ) -> List[str]:
        """Generate human-readable interpretation of logistic regression."""
        interpretations = []

        for idx in top_indices:
            name = feature_names[idx]
            c = coef[idx]
            odds = np.exp(c)

            if c > 0:
                interp = f"Higher {name} increases risk (OR={odds:.2f})"
            else:
                interp = f"Higher {name} decreases risk (OR={odds:.2f})"

            interpretations.append(interp)

        return interpretations

    def _extract_tree_rules(
        self,
        tree: DecisionTreeClassifier,
        feature_names: List[str]
    ) -> List[str]:
        """Extract decision rules from tree."""
        tree_ = tree.tree_
        rules = []

        def recurse(node, depth, rule):
            if tree_.feature[node] != -2:  # Not a leaf
                name = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]

                # Left branch (<=)
                recurse(
                    tree_.children_left[node],
                    depth + 1,
                    rule + [f"{name} <= {threshold:.2f}"]
                )
                # Right branch (>)
                recurse(
                    tree_.children_right[node],
                    depth + 1,
                    rule + [f"{name} > {threshold:.2f}"]
                )
            else:
                # Leaf node
                class_idx = np.argmax(tree_.value[node])
                confidence = tree_.value[node][0][class_idx] / tree_.value[node][0].sum()
                if len(rule) <= 3:  # Only short rules
                    rules.append(
                        f"IF {' AND '.join(rule)} THEN class={class_idx} (conf={confidence:.2f})"
                    )

        recurse(0, 0, [])
        return rules[:10]  # Return top 10 rules

    def predict(self, X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        return self.models[model_name].predict(X)

    def predict_proba(self, X: np.ndarray, model_name: str = 'random_forest') -> np.ndarray:
        """Get prediction probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        return self.models[model_name].predict_proba(X)
