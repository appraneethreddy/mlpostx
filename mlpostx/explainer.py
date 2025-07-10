import pandas as pd
from typing import Any, Dict, Tuple, Union

# Import the from-scratch implementation modules
from ._target_analyzer import analyze_target_relationships
from ._model_explainer import explain_local_from_scratch, explain_global_from_scratch
from ._visualizer import plot_global_importance
from ._text_generator import generate_local_explanation_text
from ._causal_analyzer import analyze_causal_effect


class Explainer:
    """The main interface for the MLPostX explainability toolkit."""

    def __init__(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series):
        """Initializes the Explainer.

        Args:
            model: A trained model object with a `predict` method.
            X_train: The training data features (pd.DataFrame).
            y_train: The training data target (pd.Series).
        """
        if not hasattr(model, 'predict'):
            raise ValueError("The model must have a 'predict' method.")
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self._task_type = self._determine_task_type()

    def _determine_task_type(self) -> str:
        """Determines if the task is classification or regression."""
        # Heuristic: If the number of unique values in y_train is small, it's likely classification.
        # A more robust check could be added (e.g., checking y_train.dtype).
        if self.y_train.nunique() <= 20: # A common heuristic
            print("Task Type: Classification")
            return 'classification'
        else:
            print("Task Type: Regression")
            return 'regression'

    def profile_data(self) -> Dict[str, float]:
        """Analyzes and returns the relationship between each feature and the target."""
        print("--- Generating Target-Oriented Data Explanation ---")
        return analyze_target_relationships(self.X_train, self.y_train, self._task_type)

    def explain_global(self, method: str = 'permutation_importance') -> Dict[str, Any]:
        """Generates a global explanation for the model."""
        print(f"Generating global explanation via '{method}'...")
        return explain_global_from_scratch(
            model=self.model,
            X=self.X_train,
            y=self.y_train,
            method=method,
            task_type=self._task_type
        )

    def explain_local(
        self, instance: pd.DataFrame, method: str = 'local_surrogate'
    ) -> str:
        """Generates a text-based local explanation for a single instance."""
        if not isinstance(instance, pd.DataFrame) or instance.shape[0] != 1:
            raise ValueError("The 'instance' must be a single-row pandas DataFrame.")
        
        print(f"Generating local explanation via '{method}'...")
        explanation_output = explain_local_from_scratch(
            model=self.model,
            X_train=self.X_train,
            instance=instance,
            method=method,
            task_type=self._task_type
        )

        if method == 'local_surrogate':
            importances, confidence = explanation_output
            return generate_local_explanation_text(importances, confidence)
        elif method == 'rule_based':
            rule, confidence_metrics = explanation_output
            return f"{rule}\n(Confidence: Precision={confidence_metrics['rule_precision']:.2f}, Coverage={confidence_metrics['rule_coverage']:.2f})"
        else:
            return str(explanation_output)

    def explain_causal(self, feature_to_intervene: str, intervention_value: float) -> Dict[str, float]:
        """Performs a causal intervention analysis for a given feature."""
        print(f"--- Performing Causal Analysis on '{feature_to_intervene}' ---")
        return analyze_causal_effect(
            model=self.model,
            X_train=self.X_train,
            feature_to_intervene=feature_to_intervene,
            intervention_value=intervention_value
        )

    def visualize(self, explanation_type: str, explanation_data: Dict, title: str = None):
        """Visualizes an explanation."""
        if explanation_type == 'global':
            plot_title = title if title else 'Global Feature Importance'
            plot_global_importance(explanation_data, plot_title=plot_title)

        else:
            raise NotImplementedError(f"Visualization for '{explanation_type}' is not yet implemented.")
