import pandas as pd
from typing import Any, Dict, Tuple, Union

# Import the from-scratch implementation modules
from ._data_profiler import profile_data_from_scratch
from ._model_explainer import explain_local_from_scratch, explain_global_from_scratch
from ._visualizer import plot_global_importance, plot_local_importance


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

    def profile_data(self) -> Dict[str, Any]:
        """Generates a data profile including statistics and correlations."""
        print("Generating data profile...")
        return profile_data_from_scratch(self.X_train)

    def explain_global(self, method: str = 'permutation_importance') -> Dict[str, Any]:
        """Generates a global explanation for the model."""
        print(f"Generating global explanation via '{method}'...")
        return explain_global_from_scratch(
            model=self.model,
            X=self.X_train,
            y=self.y_train,
            method=method
        )

    def explain_local(
        self, instance: pd.DataFrame, method: str = 'local_surrogate'
    ) -> Union[Tuple[Dict[str, float], float], Tuple[str, Dict[str, float]]]:
        """Generates a local explanation for a single instance."""
        if not isinstance(instance, pd.DataFrame) or instance.shape[0] != 1:
            raise ValueError("The 'instance' must be a single-row pandas DataFrame.")
        
        print(f"Generating local explanation via '{method}'...")
        return explain_local_from_scratch(
            model=self.model,
            X_train=self.X_train,
            instance=instance,
            method=method
        )

    def visualize(self, explanation_type: str, explanation_data: Dict, title: str = None):
        """Visualizes an explanation."""
        if explanation_type == 'global':
            plot_title = title if title else 'Global Feature Importance'
            plot_global_importance(explanation_data, plot_title=plot_title)
        elif explanation_type == 'local':
            if not isinstance(explanation_data, dict):
                raise TypeError("Local visualization requires a dictionary of feature importances.")
            plot_title = title if title else 'Local Feature Importance'
            plot_local_importance(explanation_data, plot_title=plot_title)
        else:
            raise NotImplementedError(f"Visualization for '{explanation_type}' is not yet implemented.")
