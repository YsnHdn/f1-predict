"""
Main Streamlit app for F1 prediction visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import sys
import glob

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supervisor import SupervisorAgent
from agents.utils.logging import setup_logger

# Setup logging
logger = setup_logger('streamlit_app')

# Set page configuration
st.set_page_config(
    page_title="F1 Prediction System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_predictions(predictions_dir="predictions"):
    """Load all available predictions from the predictions directory."""
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
            
            # Create a key for this prediction
            key = f"{timestamp}_{race_name}_{prediction_type}"
            
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

def run_prediction_workflow(race_name, circuit, race_date, prediction_types):
    """Run the prediction workflow using the supervisor agent."""
    try:
        # Initialize the supervisor agent
        supervisor = SupervisorAgent()
        
        # Prepare context for execution
        context = {
            'race_name': race_name,
            'circuit': circuit,
            'race_date': race_date,
            'prediction_types': prediction_types
        }
        
        # Execute the workflow
        results = supervisor.execute(context)
        
        return True, results
    except Exception as e:
        return False, str(e)

def display_prediction(prediction_data):
    """Display a prediction in the Streamlit UI."""
    race_info = prediction_data.get('race_info', {})
    prediction_type = prediction_data.get('prediction_type', 'Unknown')
    
    st.subheader(f"{race_info.get('name', 'Unknown Race')} - {prediction_type.title()} Prediction")
    st.write(f"**Circuit:** {race_info.get('circuit', 'Unknown')}")
    st.write(f"**Date:** {race_info.get('date', 'Unknown')}")
    st.write(f"**Prediction made:** {prediction_data.get('timestamp', 'Unknown')}")
    
    # Display predictions
    predictions = prediction_data.get('data', [])
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Highlight top 3 positions
        def highlight_top3(s):
            is_top3 = pd.Series(data=False, index=s.index)
            is_top3.iloc[:3] = True
            return ['background-color: #FFD700' if is_top3.iloc[i] and i == 0
                   else 'background-color: #C0C0C0' if is_top3.iloc[i] and i == 1
                   else 'background-color: #CD7F32' if is_top3.iloc[i] and i == 2
                   else '' for i in range(len(is_top3))]
        
        # Display results as table
        if 'Driver' in df.columns and 'PredictedPosition' in df.columns:
            # Ensure the DataFrame is sorted by predicted position
            df = df.sort_values('PredictedPosition')
            
            # Select columns to display
            display_cols = ['Driver', 'PredictedPosition']
            if 'Team' in df.columns:
                display_cols.insert(1, 'Team')
            if 'Confidence' in df.columns:
                display_cols.append('Confidence')
            
            # Display the table with highlighting
            st.dataframe(df[display_cols].style.apply(highlight_top3))
        else:
            st.write("Prediction data format not recognized")
    else:
        st.write("No prediction data available")

def main():
    """Main function for the Streamlit app."""
    st.title("🏎️ F1 Prediction System")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["View Predictions", "Generate New Prediction", "About"]
    )
    
    if page == "View Predictions":
        st.header("View Predictions")
        
        # Load available predictions
        predictions = load_predictions()
        
        if not predictions:
            st.warning("No predictions found. Generate new predictions or check the predictions directory.")
        else:
            # Create a selectbox to choose which prediction to view
            prediction_options = []
            for key, pred in predictions.items():
                race_name = pred['race_info'].get('name', 'Unknown Race')
                pred_type = pred['prediction_type'].title()
                timestamp = pred['timestamp']
                option_text = f"{race_name} - {pred_type} ({timestamp})"
                prediction_options.append((option_text, key))
            
            selected_option = st.selectbox(
                "Select a prediction to view",
                [option[0] for option in prediction_options]
            )
            
            # Find the selected prediction
            selected_key = next((opt[1] for opt in prediction_options if opt[0] == selected_option), None)
            
            if selected_key and selected_key in predictions:
                display_prediction(predictions[selected_key])
    
    elif page == "Generate New Prediction":
        st.header("Generate New Prediction")
        
        # Form for generating new predictions
        with st.form("prediction_form"):
            race_name = st.text_input("Race Name (e.g., Belgian Grand Prix)")
            circuit = st.text_input("Circuit Identifier (e.g., spa)")
            race_date = st.date_input("Race Date")
            
            # Prediction types checkboxes
            st.write("Prediction Types:")
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_pred = st.checkbox("Initial Prediction", value=True)
            with col2:
                pre_race_pred = st.checkbox("Pre-Race Prediction", value=True)
            with col3:
                race_day_pred = st.checkbox("Race Day Prediction", value=True)
            
            # Submit button
            submit_button = st.form_submit_button("Generate Predictions")
            
            if submit_button:
                # Validate inputs
                if not race_name or not circuit:
                    st.error("Race name and circuit are required")
                else:
                    # Prepare prediction types
                    prediction_types = []
                    if initial_pred:
                        prediction_types.append("initial")
                    if pre_race_pred:
                        prediction_types.append("pre_race")
                    if race_day_pred:
                        prediction_types.append("race_day")
                    
                    if not prediction_types:
                        st.error("At least one prediction type must be selected")
                    else:
                        # Show progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Run prediction workflow
                        status_text.text("Starting prediction workflow...")
                        progress_bar.progress(10)
                        
                        success, results = run_prediction_workflow(
                            race_name, 
                            circuit, 
                            race_date.strftime('%Y-%m-%d'),
                            prediction_types
                        )
                        
                        progress_bar.progress(100)
                        
                        if success:
                            st.success("Predictions generated successfully!")
                            status_text.text("Predictions complete. Go to 'View Predictions' to see results.")
                        else:
                            st.error(f"Error generating predictions: {results}")
                            status_text.text("Prediction failed. Please check the logs for details.")
    
    elif page == "About":
        st.header("About F1 Prediction System")
        
        st.markdown("""
        ### Overview
        
        The F1 Prediction System uses machine learning models to predict Formula 1 race outcomes. 
        The system generates three types of predictions throughout a race weekend:
        
        1. **Initial Prediction** - Made before qualifying, based on historical performance data
        2. **Pre-Race Prediction** - Made after qualifying, incorporating grid positions
        3. **Race Day Prediction** - Made just before the race, incorporating last-minute information like weather and tire choices
        
        ### How It Works
        
        The system uses a multi-agent architecture to:
        
        - Collect race data from the FastF1 API
        - Monitor weather conditions for the circuit
        - Generate predictions using machine learning models
        - Analyze prediction performance against actual results
        
        ### Technologies Used
        
        - **FastF1** - For F1 data collection
        - **CrewAI** - For the agent architecture
        - **Scikit-learn** - For prediction models
        - **Pandas & NumPy** - For data processing
        - **Streamlit** - For this user interface
        
        ### Contact
        
        For questions or support, please contact: y.handane@gmail.com
        """)

if __name__ == "__main__":
    main()