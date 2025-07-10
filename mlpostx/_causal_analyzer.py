import pandas as pd
from typing import Any, Dict

def analyze_causal_effect(
    model: Any, 
    X_train: pd.DataFrame, 
    feature_to_intervene: str, 
    intervention_value: float
) -> Dict[str, float]:
    """Performs a simple causal intervention analysis.

    Estimates the causal effect of a feature by setting it to a fixed value
    and observing the change in the model's average prediction.

    Args:
        model: The trained model.
        X_train: The training data features.
        feature_to_intervene: The name of the feature to change.
        intervention_value: The value to set the feature to.

    Returns:
        A dictionary containing the original and new average predictions,
        and the estimated causal effect.
    """
    if feature_to_intervene not in X_train.columns:
        raise ValueError(f"Feature '{feature_to_intervene}' not found in the dataset.")

    # 1. Calculate the original average prediction
    original_prediction = model.predict(X_train).mean()

    # 2. Create the intervened dataset
    X_intervened = X_train.copy()
    X_intervened[feature_to_intervene] = intervention_value

    # 3. Calculate the new average prediction
    intervened_prediction = model.predict(X_intervened).mean()

    # 4. Estimate the causal effect
    causal_effect = intervened_prediction - original_prediction

    return {
        'original_average_prediction': original_prediction,
        'intervened_average_prediction': intervened_prediction,
        'estimated_causal_effect': causal_effect
    }
