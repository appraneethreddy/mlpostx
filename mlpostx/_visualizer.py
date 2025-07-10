import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

def plot_global_importance(importances: Dict[str, float], plot_title: str = 'Global Feature Importance'):
    """Generates and displays a bar plot for global feature importances.

    Args:
        importances: A dictionary mapping feature names to their importance scores.
        plot_title: The title for the plot.
    """
    importance_series = pd.Series(importances).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    # Pass 'y' to 'hue' and disable legend to address seaborn FutureWarning
    sns.barplot(x=importance_series.values, y=importance_series.index, hue=importance_series.index, palette='viridis', legend=False)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title(plot_title)
    plt.tight_layout()
    plt.show()

def plot_local_importance(importances: Dict[str, float], plot_title: str = 'Local Feature Importance'):
    """Generates and displays a bar plot for local feature importances.

    Handles both positive and negative importance values to show feature contributions.

    Args:
        importances: A dictionary mapping feature names to their local importance scores.
        plot_title: The title for the plot.
    """
    importance_series = pd.Series(importances)
    # Sort by absolute value to show most impactful features, but plot actual values
    sorted_indices = importance_series.abs().sort_values(ascending=False).index
    importance_series = importance_series[sorted_indices]

    # Create a color palette based on the sign of the importance
    colors = ['#34A853' if x > 0 else '#EA4335' for x in importance_series.values]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=importance_series.values, y=importance_series.index, palette=colors, hue=importance_series.index, legend=False)
    plt.xlabel('Feature Contribution')
    plt.ylabel('Features')
    plt.title(plot_title)
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()
