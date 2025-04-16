"""
Table components for the F1 prediction visualization app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go

def create_predictions_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted table for prediction results.
    
    Args:
        predictions: DataFrame with prediction results
        
    Returns:
        Formatted DataFrame for display
    """
    # Ensure predictions are sorted by PredictedPosition
    sorted_df = predictions.sort_values('PredictedPosition')
    
    # Select relevant columns
    display_cols = ['Driver', 'PredictedPosition']
    if 'Team' in sorted_df.columns:
        display_cols.insert(1, 'Team')
    if 'Confidence' in sorted_df.columns:
        display_cols.append('Confidence')
    
    return sorted_df[display_cols]

def style_predictions_table(df: pd.DataFrame) -> pd.DataFrame.style:
    """
    Apply styling to the predictions table.
    
    Args:
        df: DataFrame with prediction results
        
    Returns:
        Styled DataFrame
    """
    # Define styling function for rows
    def highlight_top3(s):
        is_top3 = pd.Series(data=False, index=s.index)
        is_top3.iloc[:3] = True
        return ['background-color: #FFD700' if is_top3.iloc[i] and i == 0
               else 'background-color: #C0C0C0' if is_top3.iloc[i] and i == 1
               else 'background-color: #CD7F32' if is_top3.iloc[i] and i == 2
               else '' for i in range(len(is_top3))]
    
    # Apply styling
    return df.style.apply(highlight_top3)

def create_comparison_table(predictions: pd.DataFrame, actual_results: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create a table comparing predicted positions with actual results.
    
    Args:
        predictions: DataFrame with prediction results
        actual_results: Optional DataFrame with actual race results
        
    Returns:
        DataFrame with comparison
    """
    # If no actual results, just return predictions
    if actual_results is None:
        return create_predictions_table(predictions)
    
    # Merge predictions with actual results
    comparison_df = pd.merge(
        predictions[['Driver', 'PredictedPosition']], 
        actual_results[['Driver', 'Position']],
        on='Driver',
        how='inner'
    )
    
    # Add error column
    comparison_df['Error'] = comparison_df['PredictedPosition'] - comparison_df['Position']
    comparison_df['AbsError'] = np.abs(comparison_df['Error'])
    
    # Sort by actual position
    comparison_df = comparison_df.sort_values('Position')
    
    # Add Team column if available
    if 'Team' in predictions.columns:
        team_mapping = predictions.set_index('Driver')['Team'].to_dict()
        comparison_df['Team'] = comparison_df['Driver'].map(team_mapping)
        
        # Reorder columns
        comparison_df = comparison_df[['Driver', 'Team', 'Position', 'PredictedPosition', 'Error', 'AbsError']]
    else:
        comparison_df = comparison_df[['Driver', 'Position', 'PredictedPosition', 'Error', 'AbsError']]
    
    return comparison_df

def style_comparison_table(df: pd.DataFrame) -> pd.DataFrame.style:
    """
    Apply styling to the comparison table.
    
    Args:
        df: DataFrame with comparison results
        
    Returns:
        Styled DataFrame
    """
    # Define styling function for the Error column
    def highlight_errors(s):
        is_error = s.name == 'Error'
        if not is_error:
            return [''] * len(s)
        
        return ['color: red' if x > 0 else 'color: green' if x < 0 else '' for x in s]
    
    # Define styling function for top 3 actual positions
    def highlight_top3_actual(s):
        if 'Position' not in df.columns:
            return [''] * len(s)
        
        positions = df['Position']
        return ['background-color: #FFD700' if pos == 1
               else 'background-color: #C0C0C0' if pos == 2
               else 'background-color: #CD7F32' if pos == 3
               else '' for pos in positions]
    
    # Apply styling
    return df.style.apply(highlight_errors).apply(highlight_top3_actual, axis=0)

def create_metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """
    Create a table displaying prediction metrics.
    
    Args:
        metrics: Dictionary with metric names and values
        
    Returns:
        DataFrame with metrics for display
    """
    if not metrics:
        return pd.DataFrame()
    
    # Create DataFrame from metrics dictionary
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    # Format metric names for display
    metrics_df['Metric'] = metrics_df['Metric'].apply(lambda x: ' '.join(word.capitalize() for word in x.split('_')))
    
    # Format values based on type
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
    
    return metrics_df

def create_scenario_comparison_table(scenarios_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create a table comparing predictions across different scenarios.
    
    Args:
        scenarios_data: Dictionary mapping scenario names to prediction DataFrames
        
    Returns:
        DataFrame with scenario comparison
    """
    if not scenarios_data:
        return pd.DataFrame()
    
    # Get unique drivers from all scenarios
    all_drivers = set()
    for df in scenarios_data.values():
        all_drivers.update(df['Driver'].unique())
    
    # Create comparison DataFrame
    comparison_data = []
    
    for driver in sorted(all_drivers):
        row = {'Driver': driver}
        
        for scenario, df in scenarios_data.items():
            driver_data = df[df['Driver'] == driver]
            if not driver_data.empty:
                row[scenario] = int(driver_data['PredictedPosition'].iloc[0])
            else:
                row[scenario] = None
        
        comparison_data.append(row)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add Team column if available
    for df in scenarios_data.values():
        if 'Team' in df.columns:
            # Create driver to team mapping
            team_mapping = df.set_index('Driver')['Team'].to_dict()
            comparison_df['Team'] = comparison_df['Driver'].map(team_mapping)
            
            # Move Team column to be right after Driver
            cols = comparison_df.columns.tolist()
            cols.insert(1, cols.pop(cols.index('Team')))
            comparison_df = comparison_df[cols]
            
            break
    
    return comparison_df

def create_plotly_table(df: pd.DataFrame, title: str = None) -> go.Figure:
    """
    Create a Plotly table for rich interactive display.
    
    Args:
        df: DataFrame to display
        title: Optional table title
        
    Returns:
        Plotly figure with table
    """
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color='lavender',
            align='left'
        )
    )])
    
    if title:
        fig.update_layout(title=title)
    
    return fig