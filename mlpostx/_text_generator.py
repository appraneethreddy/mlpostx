from typing import Dict, List

def generate_local_explanation_text(importances: Dict[str, float], confidence: float, top_n: int = 3) -> str:
    """Generates a human-readable text summary for a local explanation.

    Args:
        importances: A dictionary of feature importances from a local surrogate.
        confidence: The confidence score (fidelity) of the explanation.
        top_n: The number of top features to include in the summary.

    Returns:
        A string containing the narrative explanation.
    """
    if not importances:
        return "No significant features found to explain this prediction."

    # Sort features by the absolute value of their importance
    sorted_features = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)

    # Separate into positive (supporting) and negative (contradicting) contributions
    positive_contributors = [(f, i) for f, i in sorted_features if i > 0]
    negative_contributors = [(f, i) for f, i in sorted_features if i < 0]

    summary_parts: List[str] = []

    # Start with the overall prediction summary
    summary_parts.append(
        f"This prediction was primarily driven by {len(sorted_features)} features. "
        f"The explanation has a confidence score (fidelity) of {confidence:.2f}."
    )

    # Describe the top positive contributors
    if positive_contributors:
        top_pos = [f for f, i in positive_contributors[:top_n]]
        summary_parts.append(f"The main factors increasing the prediction were: {', '.join(top_pos)}.")

    # Describe the top negative contributors
    if negative_contributors:
        top_neg = [f for f, i in negative_contributors[:top_n]]
        summary_parts.append(f"The main factors decreasing the prediction were: {', '.join(top_neg)}.")
    
    return "\n".join(summary_parts)
