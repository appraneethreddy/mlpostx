import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root to the Python path to allow importing mlpostx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlpostx import Explainer

def main():
    """Main function to run the demonstration."""
    # 1. Load Data
    print("--- 1. Loading Data ---")
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset loaded and split.")

    # 2. Train Model
    print("\n--- 2. Training Model ---")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("XGBoost model trained.")

    # 3. Initialize Explainer
    print("\n--- 3. Initializing Explainer ---")
    explainer = Explainer(model, X_train=X_train, y_train=y_train)
    print("Explainer initialized.")

    # 4. Profile Data
    print("\n--- 4. Generating Data Profile ---")
    profile = explainer.profile_data()
    print("\nDescriptive Statistics:")
    print(profile['descriptive_statistics'])
    print("\nCorrelation Matrix:")
    print(profile['correlation_matrix'])

    # 5. Get Global Explanation
    print("\n--- 5. Generating Global Explanation (Permutation Importance) ---")
    global_explanation = explainer.explain_global()
    sorted_global_importances = sorted(global_explanation.items(), key=lambda item: item[1], reverse=True)
    print("\nFeature Importances:")
    for feature, importance in sorted_global_importances:
        print(f"{feature}: {importance:.4f}")

    # 6. Get Local Explanation (Surrogate Model)
    print("\n--- 6. Generating Local Explanation (Local Surrogate) ---")
    instance_to_explain = X_test.iloc[[0]]
    local_explanation, confidence = explainer.explain_local(instance_to_explain, method='local_surrogate')
    sorted_local_importances = sorted(local_explanation.items(), key=lambda item: item[1], reverse=True)
    print(f"\nLocal explanation for instance 0 (Confidence/Fidelity: {confidence:.4f}):")
    for feature, importance in sorted_local_importances:
        print(f"{feature}: {importance:.4f}")

    # 7. Get Rule-Based Explanation
    print("\n--- 7. Generating Rule-Based Explanation ---")
    rule, rule_confidence = explainer.explain_local(instance_to_explain, method='rule_based')
    print(f"\nActionable Rule for instance 0:")
    print(rule)
    print(f"Rule Confidence: {rule_confidence}")

    # 8. Visualize Global Explanation
    print("\n--- 8. Visualizing Global Explanation ---")
    explainer.visualize('global', global_explanation)

    # 9. Visualize Local Explanation
    print("\n--- 9. Visualizing Local Explanation ---")
    explainer.visualize(
        'local',
        local_explanation,
        title=f"Local Explanation for Instance 0 (Fidelity: {confidence:.2f})"
    )

    print("\n--- Demonstration Complete ---")

if __name__ == '__main__':
    main()
