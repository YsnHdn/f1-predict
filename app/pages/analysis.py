"""
Analysis page for the F1 prediction visualization app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import glob
from datetime import datetime
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.components.charts import (
    create_position_comparison_chart,
    create_prediction_confidence_chart,
    create_scenario_comparison_chart
)
from app.components.tables import (
    create_comparison_table,
    style_comparison_table,
    create_metrics_table,
    create_plotly_table,
    create_scenario_comparison_table
)
from agents.utils.logging import setup_logger

# Setup logging
logger = setup_logger('analysis_page')

def load_analysis_results(analysis_dir="analysis"):
    """Load all available analysis results from the analysis directory."""
    analysis_results = {}
    
    if not os.path.exists(analysis_dir):
        return analysis_results
    
    # Find all analysis report files
    report_files = glob.glob(os.path.join(analysis_dir, "reports", "*.md"))
    metrics_files = glob.glob(os.path.join(analysis_dir, "metrics", "*.json"))
    chart_files = glob.glob(os.path.join(analysis_dir, "charts", "*.png"))
    
    # Process report files
    for file_path in report_files:
        try:
            # Extract information from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            if len(parts) >= 3:
                timestamp = parts[0]
                race_name = '_'.join(parts[1:-1])
                pred_type = parts[-1].split('.')[0]
                
                # Create a key for this analysis
                key = f"{timestamp}_{race_name}_{pred_type}"
                
                # Initialize if not exists
                if key not in analysis_results:
                    analysis_results[key] = {
                        'race_name': race_name.replace('_', ' '),
                        'prediction_type': pred_type,
                        'timestamp': timestamp,
                        'report_path': file_path,
                        'metrics_path': None,
                        'chart_paths': []
                    }
                else:
                    analysis_results[key]['report_path'] = file_path
        except Exception as e:
            logger.error(f"Error processing report file {file_path}: {str(e)}")
    
    # Process metrics files
    for file_path in metrics_files:
        try:
            # Extract information from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            if len(parts) >= 3:
                timestamp = parts[0]
                race_name = '_'.join(parts[1:-1])
                pred_type = parts[-1].split('.')[0]
                
                # Create a key for this analysis
                key = f"{timestamp}_{race_name}_{pred_type}"
                
                # Initialize if not exists
                if key not in analysis_results:
                    analysis_results[key] = {
                        'race_name': race_name.replace('_', ' '),
                        'prediction_type': pred_type,
                        'timestamp': timestamp,
                        'report_path': None,
                        'metrics_path': file_path,
                        'chart_paths': []
                    }
                else:
                    analysis_results[key]['metrics_path'] = file_path
                    
                # Load metrics
                with open(file_path, 'r') as f:
                    analysis_results[key]['metrics'] = json.load(f)
        except Exception as e:
            logger.error(f"Error processing metrics file {file_path}: {str(e)}")
    
    # Process chart files
    for file_path in chart_files:
        try:
            # Extract information from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            if len(parts) >= 4:  # timestamp_race_predtype_charttype.png
                timestamp = parts[0]
                race_name = '_'.join(parts[1:-2])
                pred_type = parts[-2]
                chart_type = parts[-1].split('.')[0]
                
                # Create a key for this analysis
                key = f"{timestamp}_{race_name}_{pred_type}"
                
                # Initialize if not exists
                if key not in analysis_results:
                    analysis_results[key] = {
                        'race_name': race_name.replace('_', ' '),
                        'prediction_type': pred_type,
                        'timestamp': timestamp,
                        'report_path': None,
                        'metrics_path': None,
                        'chart_paths': []
                    }
                
                # Add chart path
                analysis_results[key]['chart_paths'].append({
                    'type': chart_type,
                    'path': file_path
                })
        except Exception as e:
            logger.error(f"Error processing chart file {file_path}: {str(e)}")
    
    return analysis_results

def load_prediction_results(predictions_dir="predictions"):
    """Load all available prediction results from the predictions directory."""
    predictions = {}
    
    if not os.path.exists(predictions_dir):
        return predictions
    
    # Find all prediction JSON files
    json_files = glob.glob(os.path.join(predictions_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                prediction_data = json.load(f)
            
            # Extract key information
            prediction_type = prediction_data.get('prediction_type', 'unknown')
            race_info = prediction_data.get('race_info', {})
            race_name = race_info.get('name', 'Unknown Race')
            timestamp = prediction_data.get('timestamp', '')
            
            # Clean race name for key
            race_key = race_name.replace(' ', '_')
            
            # Create a key for this prediction
            key = f"{timestamp}_{race_key}_{prediction_type}"
            
            # Store prediction data
            predictions[key] = {
                'file_path': file_path,
                'prediction_type': prediction_type,
                'race_info': race_info,
                'timestamp': timestamp,
                'data': prediction_data.get('predictions', [])
            }
        except Exception as e:
            logger.error(f"Error loading prediction from {file_path}: {str(e)}")
    
    return predictions

def display_report(report_path: str):
    """Display a markdown report."""
    try:
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        st.markdown(report_content)
    except Exception as e:
        st.error(f"Error loading report: {str(e)}")

def display_metrics(metrics: Dict[str, Any]):
    """Display metrics in a formatted way."""
    if not metrics:
        st.warning("No metrics available")
        return
    
    # Create metrics table
    metrics_df = create_metrics_table(metrics)
    
    if not metrics_df.empty:
        st.table(metrics_df)
    else:
        st.warning("No metrics data to display")
    
    # Show key metrics as Streamlit metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'mae' in metrics:
            st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
        if 'winner_prediction_accuracy' in metrics:
            st.metric("Winner Prediction", f"{metrics['winner_prediction_accuracy']*100:.1f}%")
    
    with col2:
        if 'rmse' in metrics:
            st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
        if 'podium_prediction_accuracy' in metrics:
            st.metric("Podium Prediction", f"{metrics['podium_prediction_accuracy']*100:.1f}%")
    
    with col3:
        if 'exact_position_accuracy' in metrics:
            st.metric("Exact Position Accuracy", f"{metrics['exact_position_accuracy']*100:.1f}%")
        if 'within_one_accuracy' in metrics:
            st.metric("Within 1 Position", f"{metrics['within_one_accuracy']*100:.1f}%")

def display_comparison(predictions: Dict[str, Any], actual_results: Dict[str, Any]):
    """Display a comparison between predictions and actual results."""
    if not predictions or not actual_results:
        st.warning("Missing data for comparison")
        return
    
    # Convert to DataFrame if needed
    pred_df = pd.DataFrame(predictions['data']) if isinstance(predictions['data'], list) else predictions['data']
    actual_df = pd.DataFrame(actual_results['data']) if isinstance(actual_results['data'], list) else actual_results['data']
    
    # Create comparison table
    comparison_df = create_comparison_table(pred_df, actual_df)
    
    if not comparison_df.empty:
        st.write("### Position Comparison")
        st.dataframe(style_comparison_table(comparison_df))
        
        # Show visualization
        st.write("### Visual Comparison")
        comparison_chart = create_position_comparison_chart(pred_df, actual_df)
        st.plotly_chart(comparison_chart, use_container_width=True)
    else:
        st.warning("Could not create comparison - incompatible data formats")

def display_charts(analysis: Dict[str, Any], predictions: Dict[str, Any] = None):
    """Display available charts for the analysis."""
    chart_paths = analysis.get('chart_paths', [])
    
    if not chart_paths:
        # If no pre-generated charts, try to create new ones
        if predictions:
            pred_df = pd.DataFrame(predictions['data']) if isinstance(predictions['data'], list) else predictions['data']
            
            st.write("### Prediction Visualization")
            
            # Create confidence chart if available
            if 'Confidence' in pred_df.columns:
                confidence_chart = create_prediction_confidence_chart(pred_df)
                st.plotly_chart(confidence_chart, use_container_width=True)
    else:
        # Display pre-generated charts
        for chart_info in chart_paths:
            chart_type = chart_info.get('type', 'unknown')
            path = chart_info.get('path')
            
            if path and os.path.exists(path):
                st.write(f"### {chart_type.replace('_', ' ').title()} Chart")
                st.image(path)

def analysis_page():
    """Main function for the analysis page."""
    st.title("📊 Prediction Analysis")
    
    # Load analysis results
    analysis_results = load_analysis_results()
    predictions = load_prediction_results()
    
    if not analysis_results and not predictions:
        st.warning("No analysis or prediction data found. Run predictions and complete races to see analysis.")
        return
    
    # Allow user to select an analysis to view
    analysis_options = []
    
    for key, analysis in analysis_results.items():
        race_name = analysis['race_name']
        pred_type = analysis['prediction_type'].title()
        timestamp = analysis['timestamp']
        option_text = f"{race_name} - {pred_type} ({timestamp})"
        analysis_options.append((option_text, key))
    
    # If no analysis but predictions exist, offer to compare predictions
    prediction_options = []
    if not analysis_options and predictions:
        for key, pred in predictions.items():
            race_name = pred['race_info'].get('name', 'Unknown Race')
            pred_type = pred['prediction_type'].title()
            timestamp = pred['timestamp']
            option_text = f"{race_name} - {pred_type} ({timestamp})"
            prediction_options.append((option_text, key))
        
        st.info("No analysis results found, but predictions are available for review.")
    
    if analysis_options:
        st.subheader("Select Analysis to View")
        selected_option = st.selectbox(
            "Choose an analysis result",
            [opt[0] for opt in analysis_options]
        )
        
        # Find the selected analysis
        selected_key = next((opt[1] for opt in analysis_options if opt[0] == selected_option), None)
        
        if selected_key and selected_key in analysis_results:
            analysis = analysis_results[selected_key]
            
            # Display analysis overview
            st.write(f"### {analysis['race_name']} - {analysis['prediction_type'].title()} Analysis")
            st.write(f"Analysis time: {datetime.strptime(analysis['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Create tabs for different aspects of the analysis
            tabs = st.tabs(["Report", "Metrics", "Visualizations", "Comparison"])
            
            # Report tab
            with tabs[0]:
                if analysis.get('report_path') and os.path.exists(analysis['report_path']):
                    display_report(analysis['report_path'])
                else:
                    st.warning("No report available for this analysis.")
            
            # Metrics tab
            with tabs[1]:
                if 'metrics' in analysis:
                    display_metrics(analysis['metrics'])
                else:
                    st.warning("No metrics available for this analysis.")
            
            # Visualizations tab
            with tabs[2]:
                # Look for matching prediction data
                matching_prediction = None
                for key, pred in predictions.items():
                    if (pred['race_info'].get('name', '') == analysis['race_name'] and 
                        pred['prediction_type'] == analysis['prediction_type']):
                        matching_prediction = pred
                        break
                
                display_charts(analysis, matching_prediction)
            
            # Comparison tab
            with tabs[3]:
                # Find actual results and predictions for comparison
                actual_results = None
                matching_prediction = None
                
                for key, pred in predictions.items():
                    race_name = pred['race_info'].get('name', '')
                    pred_type = pred['prediction_type']
                    
                    if race_name == analysis['race_name']:
                        if pred_type == 'race_day':
                            actual_results = pred  # Use race day results as "actual" for now
                        
                        if pred_type == analysis['prediction_type']:
                            matching_prediction = pred
                
                if matching_prediction and actual_results:
                    display_comparison(matching_prediction, actual_results)
                else:
                    st.warning("Cannot display comparison - missing prediction or actual results.")
    
    elif prediction_options:
        st.subheader("Review Predictions")
        selected_option = st.selectbox(
            "Choose a prediction to review",
            [opt[0] for opt in prediction_options]
        )
        
        # Find the selected prediction
        selected_key = next((opt[1] for opt in prediction_options if opt[0] == selected_option), None)
        
        if selected_key and selected_key in predictions:
            pred = predictions[selected_key]
            
            # Display prediction overview
            st.write(f"### {pred['race_info'].get('name', 'Unknown Race')} - {pred['prediction_type'].title()} Prediction")
            st.write(f"Prediction time: {datetime.strptime(pred['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Convert to DataFrame if needed
            pred_df = pd.DataFrame(pred['data']) if isinstance(pred['data'], list) else pred['data']
            
            if not pred_df.empty:
                # Display table
                st.write("### Predicted Positions")
                st.dataframe(pred_df)
                
                # Display chart
                st.write("### Visualization")
                comparison_chart = create_position_comparison_chart(pred_df)
                st.plotly_chart(comparison_chart, use_container_width=True)
            else:
                st.warning("No prediction data available.")
    
    else:
        st.warning("No analysis or prediction data found.")

if __name__ == "__main__":
    # For testing this page directly
    analysis_page()