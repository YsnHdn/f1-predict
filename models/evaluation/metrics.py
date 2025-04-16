"""
Metrics module for evaluating F1 prediction models.
This module provides functions to calculate various performance metrics
for evaluating the accuracy of F1 race predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_prediction_metrics(actual_positions: Union[pd.Series, np.ndarray, List[int]], 
                               predicted_positions: Union[pd.Series, np.ndarray, List[int]],
                               driver_names: List[str] = None) -> Dict[str, float]:
    """
    Calculate various metrics to evaluate the accuracy of race position predictions.
    
    Args:
        actual_positions: Array-like object with actual race positions
        predicted_positions: Array-like object with predicted race positions
        driver_names: Optional list of driver names corresponding to positions
        
    Returns:
        Dictionary with calculated metrics
    """
    # Convert inputs to numpy arrays for consistency
    actual = np.asarray(actual_positions)
    predicted = np.asarray(predicted_positions)
    
    # Ensure inputs have the same shape
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: actual positions {actual.shape} != predicted positions {predicted.shape}")
    
    # Calculate standard regression metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Calculate position-specific metrics
    exact_matches = np.mean(actual == predicted)
    within_one = np.mean(np.abs(actual - predicted) <= 1)
    within_three = np.mean(np.abs(actual - predicted) <= 3)
    
    # Calculate top-N accuracy
    top1_actual = np.where(actual == 1)[0]
    top1_predicted = np.where(predicted == 1)[0]
    top1_accuracy = len(np.intersect1d(top1_actual, top1_predicted)) / len(top1_actual) if len(top1_actual) > 0 else 0
    
    top3_actual = np.array([i for i, pos in enumerate(actual) if pos <= 3])
    top3_predicted = np.array([i for i, pos in enumerate(predicted) if pos <= 3])
    top3_accuracy = len(np.intersect1d(top3_actual, top3_predicted)) / len(top3_actual) if len(top3_actual) > 0 else 0
    
    # Calculate podium order accuracy
    podium_correct = False
    if len(top3_actual) >= 3 and len(top3_predicted) >= 3:
        actual_podium_indices = sorted(top3_actual, key=lambda i: actual[i])
        predicted_podium_indices = sorted(top3_predicted, key=lambda i: predicted[i])
        podium_correct = np.array_equal(actual_podium_indices[:3], predicted_podium_indices[:3])
    
    # Compile metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'exact_position_accuracy': exact_matches,
        'within_one_accuracy': within_one,
        'within_three_accuracy': within_three,
        'winner_prediction_accuracy': top1_accuracy,
        'podium_prediction_accuracy': top3_accuracy,
        'podium_order_correct': float(podium_correct)  # Convert bool to float for JSON compatibility
    }
    
    return metrics

def calculate_metrics_from_dataframes(actual_df: pd.DataFrame, predicted_df: pd.DataFrame, 
                                     driver_col: str = 'Driver', 
                                     actual_pos_col: str = 'Position',
                                     pred_pos_col: str = 'PredictedPosition') -> Dict[str, float]:
    """
    Calculate prediction metrics from DataFrames containing actual and predicted positions.
    
    Args:
        actual_df: DataFrame with actual race results
        predicted_df: DataFrame with predicted race results
        driver_col: Name of the driver column in both DataFrames
        actual_pos_col: Name of the position column in actual_df
        pred_pos_col: Name of the predicted position column in predicted_df
        
    Returns:
        Dictionary with calculated metrics
    """
    # Merge on driver column
    merged_df = pd.merge(
        actual_df[[driver_col, actual_pos_col]],
        predicted_df[[driver_col, pred_pos_col]],
        on=driver_col,
        how='inner'
    )
    
    # Check if we have matching drivers
    if merged_df.empty:
        raise ValueError("No matching drivers found between actual and predicted results")
    
    # Extract position arrays
    actual_positions = merged_df[actual_pos_col].values
    predicted_positions = merged_df[pred_pos_col].values
    driver_names = merged_df[driver_col].values
    
    # Calculate metrics
    return calculate_prediction_metrics(actual_positions, predicted_positions, driver_names)

def evaluate_multiple_predictions(actual_results: pd.DataFrame, 
                                predictions: Dict[str, pd.DataFrame],
                                driver_col: str = 'Driver',
                                actual_pos_col: str = 'Position',
                                pred_pos_col: str = 'PredictedPosition') -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple prediction models against actual results.
    
    Args:
        actual_results: DataFrame with actual race results
        predictions: Dictionary mapping prediction model names to prediction DataFrames
        driver_col: Name of the driver column
        actual_pos_col: Name of the position column in actual_results
        pred_pos_col: Name of the predicted position column in prediction DataFrames
        
    Returns:
        Dictionary mapping prediction model names to metric dictionaries
    """
    results = {}
    
    for model_name, pred_df in predictions.items():
        try:
            metrics = calculate_metrics_from_dataframes(
                actual_results, pred_df, driver_col, actual_pos_col, pred_pos_col
            )
            results[model_name] = metrics
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return results

def find_best_prediction_model(evaluation_results: Dict[str, Dict[str, float]], 
                             primary_metric: str = 'mae',
                             higher_is_better: bool = False) -> Tuple[str, Dict[str, float]]:
    """
    Find the best prediction model based on a specific metric.
    
    Args:
        evaluation_results: Dictionary mapping model names to metric dictionaries
        primary_metric: Metric to use for comparison
        higher_is_better: Whether higher values of the metric indicate better performance
        
    Returns:
        Tuple of (best_model_name, best_model_metrics)
    """
    if not evaluation_results:
        raise ValueError("No evaluation results provided")
    
    # Filter out models with errors
    valid_models = {name: metrics for name, metrics in evaluation_results.items() 
                   if primary_metric in metrics and "error" not in metrics}
    
    if not valid_models:
        raise ValueError(f"No models with valid '{primary_metric}' metric found")
    
    # Find best model
    if higher_is_better:
        best_model = max(valid_models.items(), key=lambda x: x[1][primary_metric])
    else:
        best_model = min(valid_models.items(), key=lambda x: x[1][primary_metric])
    
    return best_model

def format_metrics_report(metrics: Dict[str, float], model_name: str = None) -> str:
    """
    Format metrics as a readable report string.
    
    Args:
        metrics: Dictionary with metric values
        model_name: Optional name of the model
        
    Returns:
        Formatted report string
    """
    header = f"Performance Metrics for {model_name}" if model_name else "Performance Metrics"
    report = [header, "=" * len(header), ""]
    
    # Format each metric
    for metric, value in metrics.items():
        # Format percentage metrics
        if metric in ['exact_position_accuracy', 'within_one_accuracy', 
                      'within_three_accuracy', 'winner_prediction_accuracy',
                      'podium_prediction_accuracy']:
            report.append(f"{metric.replace('_', ' ').title()}: {value:.2f} ({value*100:.1f}%)")
        # Format boolean metrics
        elif metric == 'podium_order_correct':
            report.append(f"{metric.replace('_', ' ').title()}: {'Yes' if value > 0.5 else 'No'}")
        # Format other metrics
        else:
            report.append(f"{metric.upper()}: {value:.4f}")
    
    return "\n".join(report)