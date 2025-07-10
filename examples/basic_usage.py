import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlpostx import Explainer

def run_regression_demo():
    """Demonstrates the toolkit's capabilities on a regression task."""
    print("\n" + "="*60)
    print("               RUNNING REGRESSION DEMO (DIABETES DATASET)")
    print("="*60 + "\n")

    # 1. Load Data
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 3. Initialize Explainer
    explainer = Explainer(model, X_train=X_train, y_train=y_train)

    # 4. Get Target-Oriented Data Explanation
    print("--- 1. Target-Oriented Data Explanation ---")
    target_analysis = explainer.profile_data()
    print("Feature importance based on Mutual Information with the target:")
    for feature, score in sorted(target_analysis.items(), key=lambda item: item[1], reverse=True):
        print(f"- {feature}: {score:.4f}")

    # 5. Get Global Explanation
    print("\n--- 2. Global Model Explanation (Permutation Importance) ---")
    global_explanation = explainer.explain_global()
    explainer.visualize('global', global_explanation, title="Global Feature Importance (Regression)")

    # 6. Get Text-Based Local Explanation
    print("\n--- 3. Text-Based Local Explanation ---")
    instance = X_test.iloc[[1]]
    local_text_explanation = explainer.explain_local(instance)
    print(f"Explanation for prediction on instance 1:\n{local_text_explanation}")

    # 7. Get Causal Explanation
    print("\n--- 4. Causal Intervention Analysis ---")
    causal_effect = explainer.explain_causal(feature_to_intervene='bmi', intervention_value=0.05)
    print(f"Causal effect of setting 'bmi' to 0.05: {causal_effect['estimated_causal_effect']:.4f}")

def run_classification_demo():
    """Demonstrates the toolkit's capabilities on a classification task."""
    print("\n" + "="*60)
    print("            RUNNING CLASSIFICATION DEMO (BREAST CANCER DATASET)")
    print("="*60 + "\n")

    # 1. Load Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train Model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # 3. Initialize Explainer (will automatically detect 'classification')
    explainer = Explainer(model, X_train=X_train, y_train=y_train)

    # 4. Get Target-Oriented Data Explanation
    print("--- 1. Target-Oriented Data Explanation ---")
    target_analysis = explainer.profile_data()
    print("Feature importance based on Mutual Information with the target:")
    for feature, score in sorted(target_analysis.items(), key=lambda item: item[1], reverse=True):
        print(f"- {feature}: {score:.4f}")

    # 5. Get Global Explanation
    print("\n--- 2. Global Model Explanation (Permutation Importance) ---")
    global_explanation = explainer.explain_global()
    explainer.visualize('global', global_explanation, title="Global Feature Importance (Classification)")

    # 6. Get Text-Based Local Explanation
    print("\n--- 3. Text-Based Local Explanation ---")
    instance = X_test.iloc[[1]]
    local_text_explanation = explainer.explain_local(instance)
    print(f"Explanation for prediction on instance 1:\n{local_text_explanation}")

if __name__ == '__main__':
    run_regression_demo()
    run_classification_demo()
    print("\n--- All Demonstrations Complete ---")

