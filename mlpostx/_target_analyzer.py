import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import Dict

def analyze_target_relationships(X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, float]:
    """Analyzes the relationship between each feature and the target variable.

    Args:
        X: The feature data.
        y: The target data.
        task_type: 'classification' or 'regression'.

    Returns:
        A dictionary mapping each feature to its importance score relative to the target.
    """
    print(f"Analyzing target relationships for {task_type}...")
    if task_type == 'classification':
        # Use Mutual Information for classification, as it captures any kind of relationship.
        importances = mutual_info_classif(X, y, random_state=42)
    elif task_type == 'regression':
        # For regression, we can also use Mutual Information.
        importances = mutual_info_regression(X, y, random_state=42)
    else:
        raise ValueError("Task type must be 'classification' or 'regression'.")

    return dict(zip(X.columns, importances))
