import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, export_text
from typing import Any, Dict, Tuple, Union

# --- Global Explanation Methods ---

def _calculate_permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Calculates permutation feature importance from scratch."""
    original_score = r2_score(y, model.predict(X))
    importances = {}
    for col in X.columns:
        X_permuted = X.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col])
        permuted_score = r2_score(y, model.predict(X_permuted))
        importances[col] = original_score - permuted_score
    return importances

def explain_global_from_scratch(model: Any, X: pd.DataFrame, y: pd.Series, method: str) -> Dict[str, float]:
    """Dispatcher for from-scratch global explanation methods."""
    if method == 'permutation_importance':
        return _calculate_permutation_importance(model, X, y)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

# --- Local Explanation Methods ---

def _generate_neighborhood(instance: pd.DataFrame, X_train: pd.DataFrame, n_samples: int = 1000) -> pd.DataFrame:
    """Generates a neighborhood of perturbed samples around the instance."""
    # Simple perturbation: sample from the training data's distribution
    indices = np.random.choice(X_train.index, n_samples, replace=True)
    neighborhood = X_train.loc[indices].copy()
    return neighborhood

def _explain_local_surrogate(model: Any, X_train: pd.DataFrame, instance: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """Explains an instance using a local surrogate model (Ridge)."""
    neighborhood = _generate_neighborhood(instance, X_train)
    model_predictions = model.predict(neighborhood)

    surrogate = Ridge(alpha=1.0)
    surrogate.fit(neighborhood, model_predictions)

    # Calculate local fidelity (confidence)
    surrogate_predictions = surrogate.predict(neighborhood)
    fidelity = r2_score(model_predictions, surrogate_predictions)

    # Feature importances from the surrogate model's coefficients
    importances = dict(zip(X_train.columns, surrogate.coef_))
    return importances, fidelity

def _explain_rule_based(model: Any, X_train: pd.DataFrame, instance: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """Generates a simple IF-THEN rule explanation."""
    neighborhood = _generate_neighborhood(instance, X_train, n_samples=500)
    model_predictions = model.predict(neighborhood)

    surrogate_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
    surrogate_tree.fit(neighborhood, model_predictions)

    # Get the decision path for the instance
    feature_names = X_train.columns.tolist()
    rule_text = export_text(surrogate_tree, feature_names=feature_names, decimals=2)

    # Extract the specific rule for the instance
    decision_path = surrogate_tree.decision_path(instance).indices
    node_index = surrogate_tree.apply(instance)[0]
    
    # Rule confidence metrics
    leaf_samples_mask = surrogate_tree.apply(neighborhood) == node_index
    rule_precision = model_predictions[leaf_samples_mask].mean()
    rule_coverage = leaf_samples_mask.sum() / len(neighborhood)
    
    confidence = {
        'rule_precision': rule_precision,
        'rule_coverage': rule_coverage
    }

    # Simplify the rule text to only the path taken
    instance_rule = []
    for node_id in decision_path:
        if surrogate_tree.tree_.children_left[node_id] == surrogate_tree.tree_.children_right[node_id]: # is leaf
            continue
        feature_index = surrogate_tree.tree_.feature[node_id]
        threshold = surrogate_tree.tree_.threshold[node_id]
        feature_name = feature_names[feature_index]
        # Determine if the instance went left or right
        if instance.iloc[0, feature_index] <= threshold:
            condition = f"{feature_name} <= {threshold:.2f}"
        else:
            condition = f"{feature_name} > {threshold:.2f}"
        instance_rule.append(condition)

    return f"IF {' AND '.join(instance_rule)}", confidence

def explain_local_from_scratch(
    model: Any, X_train: pd.DataFrame, instance: pd.DataFrame, method: str
) -> Union[Tuple[Dict[str, float], float], Tuple[str, Dict[str, float]]]:
    """Dispatcher for from-scratch local explanation methods."""
    if method == 'local_surrogate':
        return _explain_local_surrogate(model, X_train, instance)
    elif method == 'rule_based':
        return _explain_rule_based(model, X_train, instance)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")
