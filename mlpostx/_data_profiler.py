import pandas as pd
from typing import Dict, Any

def profile_data_from_scratch(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a basic data profile from scratch.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        A dictionary containing descriptive statistics and the correlation matrix.
    """
    # Calculate descriptive statistics
    descriptive_stats = data.describe()
    
    # Calculate correlation matrix
    correlation_matrix = data.corr(numeric_only=True)
    
    profile = {
        'descriptive_statistics': descriptive_stats,
        'correlation_matrix': correlation_matrix
    }
    
    return profile
