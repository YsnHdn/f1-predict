"""
Chart components for the F1 prediction visualization app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

def create_prediction_positions_chart(predictions: pd.DataFrame) -> alt.Chart:
    """
    Create a bar chart showing predicted positions.
    
    Args:
        predictions: DataFrame with prediction results
        
    Returns:
        Altair chart object
    """
    # Ensure predictions are sorted by PredictedPosition
    sorted_df = predictions.sort_values('PredictedPosition')
    
    # Create chart
    chart = alt.Chart(sorted_df).mark_bar().encode(
        x=alt.X('PredictedPosition:Q', title='Predicted Position'),
        y=alt.Y('Driver:N', title='Driver', sort=None),
        color=alt.condition(
            alt.datum.PredictedPosition <= 3,
            alt.value('#FFD700'),  # Gold for top 3
            alt.value('#1E90FF')   # Blue for others
        ),
        tooltip=['Driver', 'PredictedPosition', 'Team']
    ).properties(
        title='Predicted Race Positions',
        width=600,
        height=400
    )
    
    return chart

def create_position_comparison_chart(predictions: pd.DataFrame, actual_results: pd.DataFrame = None) -> go.Figure:
    """
    Create a chart comparing predicted positions with actual results.
    
    Args:
        predictions: DataFrame with prediction results
        actual_results: Optional DataFrame with actual race results
        
    Returns:
        Plotly figure object
    """
    # If no actual results, just show predicted positions
    if actual_results is None:
        fig = px.bar(
            predictions.sort_values('PredictedPosition'), 
            x='Driver', 
            y='PredictedPosition',
            title='Predicted Positions',
            labels={'PredictedPosition': 'Position', 'Driver': 'Driver'},
            color='PredictedPosition',
            color_continuous_scale='blues_r'  # Reversed blues (darker = better position)
        )
        
        # Invert y-axis so that position 1 is at the top
        fig.update_layout(yaxis={'autorange': 'reversed'})
        
        return fig
    
    # If we have actual results, merge them
    comparison_df = pd.merge(
        predictions[['Driver', 'PredictedPosition']], 
        actual_results[['Driver', 'Position']],
        on='Driver',
        how='inner'
    )
    
    # Sort by actual position
    comparison_df = comparison_df.sort_values('Position')
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for actual positions
    fig.add_trace(go.Bar(
        x=comparison_df['Driver'],
        y=comparison_df['Position'],
        name='Actual Position',
        marker_color='blue'
    ))
    
    # Add bars for predicted positions
    fig.add_trace(go.Bar(
        x=comparison_df['Driver'],
        y=comparison_df['PredictedPosition'],
        name='Predicted Position',
        marker_color='red'
    ))
    
    # Update layout
    fig.update_layout(
        title='Predicted vs Actual Positions',
        xaxis_title='Driver',
        yaxis_title='Position',
        yaxis={'autorange': 'reversed'},  # Position 1 at the top
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_team_performance_chart(predictions: pd.DataFrame) -> alt.Chart:
    """
    Create a chart showing predicted performance by team.
    
    Args:
        predictions: DataFrame with prediction results
        
    Returns:
        Altair chart object
    """
    # Check if Team column exists
    if 'Team' not in predictions.columns:
        return None
    
    # Group by team and calculate average predicted position
    team_performance = predictions.groupby('Team')['PredictedPosition'].mean().reset_index()
    team_performance = team_performance.sort_values('PredictedPosition')
    
    # Create chart
    chart = alt.Chart(team_performance).mark_bar().encode(
        x=alt.X('PredictedPosition:Q', title='Average Predicted Position'),
        y=alt.Y('Team:N', title='Team', sort=None),
        color=alt.Color('Team:N', legend=None),
        tooltip=['Team', 'PredictedPosition']
    ).properties(
        title='Team Performance Prediction',
        width=600,
        height=300
    )
    
    return chart

def create_prediction_confidence_chart(predictions: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing prediction confidence.
    
    Args:
        predictions: DataFrame with prediction results
        
    Returns:
        Plotly figure object
    """
    # Check if Confidence column exists
    if 'Confidence' not in predictions.columns:
        return None
    
    # Sort by predicted position
    sorted_df = predictions.sort_values('PredictedPosition')
    
    # Create figure
    fig = px.bar(
        sorted_df,
        x='Driver',
        y='Confidence',
        title='Prediction Confidence by Driver',
        labels={'Confidence': 'Confidence Score', 'Driver': 'Driver'},
        color='Confidence',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title='Driver',
        yaxis_title='Confidence',
        yaxis_range=[0, 1]
    )
    
    return fig

def create_scenario_comparison_chart(scenarios_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create a chart comparing predictions across different scenarios.
    
    Args:
        scenarios_data: Dictionary mapping scenario names to prediction DataFrames
        
    Returns:
        Plotly figure object
    """
    if not scenarios_data:
        return None
    
    # Combine data from all scenarios
    combined_data = []
    
    for scenario_name, df in scenarios_data.items():
        scenario_df = df[['Driver', 'PredictedPosition']].copy()
        scenario_df['Scenario'] = scenario_name
        combined_data.append(scenario_df)
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Get unique drivers and scenarios
    drivers = combined_df['Driver'].unique()
    scenarios = combined_df['Scenario'].unique()
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each scenario
    for scenario in scenarios:
        scenario_data = combined_df[combined_df['Scenario'] == scenario]
        # Sort by Driver to ensure consistent ordering
        scenario_data = scenario_data.sort_values('Driver')
        
        fig.add_trace(go.Bar(
            x=scenario_data['Driver'],
            y=scenario_data['PredictedPosition'],
            name=scenario
        ))
    
    # Update layout
    fig.update_layout(
        title='Predicted Positions Across Scenarios',
        xaxis_title='Driver',
        yaxis_title='Predicted Position',
        yaxis={'autorange': 'reversed'},  # Position 1 at the top
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig