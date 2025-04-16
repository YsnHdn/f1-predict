"""
Prediction agent for F1 prediction project.
This agent is responsible for generating race predictions using trained models.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from crewai import Task

from agents.base_agent import F1BaseAgent
from agents.utils.logging import AgentLogger
from models.initial_model import F1InitialModel
from models.pre_race_model import F1PreRaceModel
from models.race_day_model import F1RaceDayModel
from preprocessing.data_cleaning import F1DataCleaner
from preprocessing.feature_engineering import F1FeatureEngineer

# Configure logging
logger = logging.getLogger(__name__)

class PredictionAgent(F1BaseAgent):
    """
    Agent responsible for generating F1 race predictions.
    Uses machine learning models to predict race outcomes
    based on historical data, qualifying results, and race day information.
    """
    
    def __init__(self, model_dir: str = "models/saved", 
                output_dir: str = "predictions"):
        """
        Initialize the prediction agent.
        
        Args:
            model_dir: Directory containing saved models
            output_dir: Directory to store prediction results
        """
        super().__init__(
            name="Prediction Agent",
            description="Generates Formula 1 race outcome predictions using machine learning models",
            goal="Provide accurate predictions for race outcomes at different stages of a race weekend"
        )
        
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.agent_logger = AgentLogger(agent_name="PredictionAgent")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize data processor components
        self.data_cleaner = F1DataCleaner()
        self.feature_engineer = F1FeatureEngineer(scale_features=True)
        
        # Initialize models (will be loaded when needed)
        self.initial_model = F1InitialModel(estimator='rf', target='Position')
        self.pre_race_model = F1PreRaceModel(estimator='rf', target='Position')
        self.race_day_model = F1RaceDayModel(estimator='voting', target='Position')
        
        # Link models for ensemble predictions
        self.pre_race_model.initial_model = self.initial_model
        self.race_day_model.initial_model = self.initial_model
        self.race_day_model.pre_race_model = self.pre_race_model
        
        # Load models if available
        self._load_models()
        
        self.agent_logger.info(f"Initialized PredictionAgent with output directory: {output_dir}")
    
    def get_backstory(self) -> str:
        """
        Get the agent's backstory for CrewAI.
        
        Returns:
            String containing the agent's backstory
        """
        return (
            "I am a Formula 1 prediction expert with a deep understanding of racing analytics. "
            "With backgrounds in both data science and motorsport, I specialize in building "
            "and deploying machine learning models that can accurately predict race outcomes. "
            "I understand the complex factors that influence race results - from car performance "
            "and driver skill to weather conditions and race strategy. My goal is to provide "
            "the most accurate predictions possible at each stage of a race weekend, continuously "
            "refining my forecasts as new information becomes available."
        )
    
    def _load_models(self) -> None:
        """
        Load saved models from disk if available.
        """
        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            self.agent_logger.warning(f"Model directory {self.model_dir} did not exist. Created it.")
            return
        
        # Try to load initial model
        initial_model_path = os.path.join(self.model_dir, "initial_model.pkl")
        if os.path.exists(initial_model_path):
            try:
                self.initial_model.load(initial_model_path)
                self.agent_logger.info(f"Loaded initial model from {initial_model_path}")
            except Exception as e:
                self.agent_logger.error(f"Error loading initial model: {str(e)}")
        
        # Try to load pre-race model
        pre_race_model_path = os.path.join(self.model_dir, "pre_race_model.pkl")
        if os.path.exists(pre_race_model_path):
            try:
                self.pre_race_model.load(pre_race_model_path)
                self.agent_logger.info(f"Loaded pre-race model from {pre_race_model_path}")
            except Exception as e:
                self.agent_logger.error(f"Error loading pre-race model: {str(e)}")
        
        # Try to load race day model
        race_day_model_path = os.path.join(self.model_dir, "race_day_model.pkl")
        if os.path.exists(race_day_model_path):
            try:
                self.race_day_model.load(race_day_model_path)
                self.agent_logger.info(f"Loaded race day model from {race_day_model_path}")
            except Exception as e:
                self.agent_logger.error(f"Error loading race day model: {str(e)}")
    
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the prediction process.
        
        Args:
            context: Context with information needed for prediction
                Expected keys:
                - prediction_type: Type of prediction to make ('initial', 'pre_race', 'race_day')
                - data_paths: Paths to data files needed for prediction
                - race_info: Information about the race (name, date, circuit)
                - weather_conditions: Weather conditions for the race (if available)
                
        Returns:
            Dictionary with prediction results
        """
        if context is None:
            context = {}
        
        # Extract parameters from context
        prediction_type = context.get('prediction_type', 'initial')
        data_paths = context.get('data_paths', {})
        race_info = context.get('race_info', {})
        weather_conditions = context.get('weather_conditions', {})
        
        # Validate inputs
        if not data_paths:
            self.agent_logger.error("No data paths provided for prediction")
            raise ValueError("Data paths are required for prediction")
        
        # Ensure we have race info
        if not race_info:
            race_info = {
                'name': 'Unknown Grand Prix',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'circuit': 'unknown'
            }
        
        self.agent_logger.info(f"Starting {prediction_type} prediction for {race_info.get('name', 'unknown race')}")
        
        # Store results
        results = {
            'prediction_type': prediction_type,
            'race_info': race_info,
            'prediction_time': datetime.now().isoformat(),
            'predictions': None,
            'prediction_path': None,
            'confidence': None,
            'model_info': None,
            'scenarios': {}
        }
        
        try:
            # Load and prepare data based on prediction type
            if prediction_type == 'initial':
                # Initial prediction (before qualifying)
                predictions = self._make_initial_prediction(data_paths, race_info, weather_conditions)
                results['predictions'] = predictions
                
            elif prediction_type == 'pre_race':
                # Pre-race prediction (after qualifying)
                predictions = self._make_pre_race_prediction(data_paths, race_info, weather_conditions)
                results['predictions'] = predictions
                
            elif prediction_type == 'race_day':
                # Race day prediction (just before race)
                predictions = self._make_race_day_prediction(data_paths, race_info, weather_conditions)
                results['predictions'] = predictions
                
                # Also generate scenario predictions
                scenarios = self._generate_scenario_predictions(data_paths, race_info, weather_conditions)
                results['scenarios'] = scenarios
                
            else:
                self.agent_logger.error(f"Unknown prediction type: {prediction_type}")
                raise ValueError(f"Unknown prediction type: {prediction_type}")
            
            # Save prediction results
            if predictions is not None:
                prediction_path = self._save_prediction(predictions, prediction_type, race_info)
                results['prediction_path'] = prediction_path
            
            # Add model information
            if prediction_type == 'initial' and self.initial_model.is_trained:
                results['model_info'] = {
                    'name': self.initial_model.name,
                    'version': self.initial_model.version,
                    'training_date': self.initial_model.training_date.isoformat() if self.initial_model.training_date else None
                }
            elif prediction_type == 'pre_race' and self.pre_race_model.is_trained:
                results['model_info'] = {
                    'name': self.pre_race_model.name,
                    'version': self.pre_race_model.version,
                    'training_date': self.pre_race_model.training_date.isoformat() if self.pre_race_model.training_date else None
                }
            elif prediction_type == 'race_day' and self.race_day_model.is_trained:
                results['model_info'] = {
                    'name': self.race_day_model.name,
                    'version': self.race_day_model.version,
                    'training_date': self.race_day_model.training_date.isoformat() if self.race_day_model.training_date else None
                }
            
            # Publish prediction completed event
            self.publish_event("prediction_completed", {
                "prediction_type": prediction_type,
                "race_name": race_info.get('name', 'Unknown'),
                "circuit": race_info.get('circuit', 'Unknown'),
                "prediction_path": results.get('prediction_path'),
                "predictions_summary": self._get_predictions_summary(predictions)
            })
            
            return results
            
        except Exception as e:
            self.agent_logger.error(f"Error during prediction: {str(e)}")
            
            # Publish prediction failed event
            self.publish_event("prediction_failed", {
                "prediction_type": prediction_type,
                "race_name": race_info.get('name', 'Unknown'),
                "error": str(e)
            })
            
            raise
    
    def _make_initial_prediction(self, data_paths: Dict[str, str], race_info: Dict[str, Any], 
                               weather_conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Make initial prediction using the initial model (before qualifying).
        
        Args:
            data_paths: Paths to data files
            race_info: Information about the race
            weather_conditions: Weather conditions for the race
            
        Returns:
            DataFrame with prediction results
        """
        self.agent_logger.task_start("Making initial prediction")
        
        try:
            # Load historical data
            historical_data = None
            if 'historical_combined' in data_paths:
                historical_data = pd.read_csv(data_paths['historical_combined'])
                self.agent_logger.info(f"Loaded historical data from {data_paths['historical_combined']}")
            
            # Load current driver/constructor standings
            driver_standings = None
            if 'driver_standings' in data_paths:
                driver_standings = pd.read_csv(data_paths['driver_standings'])
                self.agent_logger.info(f"Loaded driver standings from {data_paths['driver_standings']}")
            
            constructor_standings = None
            if 'constructor_standings' in data_paths:
                constructor_standings = pd.read_csv(data_paths['constructor_standings'])
                self.agent_logger.info(f"Loaded constructor standings from {data_paths['constructor_standings']}")
            
            # If we don't have all necessary data, return empty result
            if historical_data is None or driver_standings is None:
                self.agent_logger.error("Missing required data for initial prediction")
                return pd.DataFrame()
            
            # Clean and prepare data
            cleaned_historical = self.data_cleaner.standardize_driver_names(historical_data)
            cleaned_historical = self.data_cleaner.standardize_team_names(cleaned_historical)
            cleaned_historical = self.data_cleaner.standardize_circuit_names(cleaned_historical, 'TrackName')
            
            # Prepare prediction data (current drivers and teams)
            prediction_data = pd.DataFrame({
                'Driver': driver_standings['Driver'],
                'Team': driver_standings['Team'] if 'Team' in driver_standings.columns else None,
                'TrackName': race_info.get('circuit', 'unknown'),
                'Year': datetime.now().year,
                'Date': race_info.get('date', datetime.now().strftime('%Y-%m-%d'))
            })
            
            # Add team info if missing
            if 'Team' not in prediction_data.columns or prediction_data['Team'].isnull().any():
                if constructor_standings is not None:
                    # Create mapping from driver to team
                    driver_to_team = {}
                    for _, row in constructor_standings.iterrows():
                        team = row['Team']
                        if 'Drivers' in row:
                            drivers = row['Drivers'].split(',')
                            for driver in drivers:
                                driver_to_team[driver.strip()] = team
                    
                    # Apply mapping
                    prediction_data['Team'] = prediction_data['Driver'].map(
                        lambda x: driver_to_team.get(x, 'Unknown')
                    )
            
            # Add weather conditions if available
            if weather_conditions:
                for key, value in weather_conditions.items():
                    prediction_data[key] = value
            
            # Generate features
            prediction_features = self.feature_engineer.create_all_features(
                prediction_data, cleaned_historical, encode_categorical=True
            )
            
            # Make prediction
            if not self.initial_model.is_trained:
                self.agent_logger.warning("Initial model is not trained. Training on historical data...")
                # Prepare historical data for training
                historical_features = self.feature_engineer.create_all_features(
                    cleaned_historical, encode_categorical=True
                )
                # Train model
                self.initial_model.train(
                    historical_features, cleaned_historical['Position']
                )
                # Save model
                self.initial_model.save(os.path.join(self.model_dir, "initial_model.pkl"))
            
            # Make predictions
            predictions = self.initial_model.predict_race_results(prediction_features)
            
            self.agent_logger.task_complete("Making initial prediction")
            return predictions
            
        except Exception as e:
            self.agent_logger.task_fail("Making initial prediction", str(e))
            raise
    
    def _make_pre_race_prediction(self, data_paths: Dict[str, str], race_info: Dict[str, Any], 
                                weather_conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Make pre-race prediction using the pre-race model (after qualifying).
        
        Args:
            data_paths: Paths to data files
            race_info: Information about the race
            weather_conditions: Weather conditions for the race
            
        Returns:
            DataFrame with prediction results
        """
        self.agent_logger.task_start("Making pre-race prediction")
        
        try:
            # Load historical data
            historical_data = None
            if 'historical_combined' in data_paths:
                historical_data = pd.read_csv(data_paths['historical_combined'])
                self.agent_logger.info(f"Loaded historical data from {data_paths['historical_combined']}")
            
            # Load qualifying data
            qualifying_data = None
            # Look for qualifying data in data_paths
            for key, path in data_paths.items():
                if '_Q' in key:
                    qualifying_data = pd.read_csv(path)
                    self.agent_logger.info(f"Loaded qualifying data from {path}")
                    break
            
            # If we don't have all necessary data, return empty result
            if historical_data is None or qualifying_data is None:
                self.agent_logger.error("Missing required data for pre-race prediction")
                return pd.DataFrame()
            
            # Clean and prepare data
            cleaned_historical = self.data_cleaner.standardize_driver_names(historical_data)
            cleaned_historical = self.data_cleaner.standardize_team_names(cleaned_historical)
            cleaned_historical = self.data_cleaner.standardize_circuit_names(cleaned_historical, 'TrackName')
            
            cleaned_qualifying = self.data_cleaner.standardize_driver_names(qualifying_data)
            cleaned_qualifying = self.data_cleaner.standardize_team_names(cleaned_qualifying)
            cleaned_qualifying = self.data_cleaner.standardize_circuit_names(cleaned_qualifying, 'TrackName')
            
            # Add grid position features and qualifying features
            qualifying_with_features = self.feature_engineer.create_grid_position_features(cleaned_qualifying)
            qualifying_with_features = self.feature_engineer.create_qualifying_features(qualifying_with_features)
            
            # Add circuit and weather information
            qualifying_with_features['TrackName'] = race_info.get('circuit', 'unknown')
            qualifying_with_features['Year'] = datetime.now().year
            qualifying_with_features['Date'] = race_info.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Add weather conditions if available
            if weather_conditions:
                for key, value in weather_conditions.items():
                    qualifying_with_features[key] = value
            
            # Generate features
            prediction_features = self.feature_engineer.create_all_features(
                qualifying_with_features, cleaned_historical, encode_categorical=True
            )
            
            # Make prediction
            if not self.pre_race_model.is_trained:
                self.agent_logger.warning("Pre-race model is not trained. Training on historical data...")
                # Prepare historical data for training
                historical_features = self.feature_engineer.create_all_features(
                    cleaned_historical, encode_categorical=True
                )
                # Train model
                self.pre_race_model.train(
                    historical_features, cleaned_historical['Position']
                )
                # Save model
                self.pre_race_model.save(os.path.join(self.model_dir, "pre_race_model.pkl"))
            
            # Make predictions
            # If initial model is trained, use blending
            if self.initial_model.is_trained:
                predictions = self.pre_race_model.blend_with_initial_model(prediction_features)
            else:
                predictions = self.pre_race_model.predict_race_results(prediction_features)
            
            self.agent_logger.task_complete("Making pre-race prediction")
            return predictions
            
        except Exception as e:
            self.agent_logger.task_fail("Making pre-race prediction", str(e))
            raise
    
    def _make_race_day_prediction(self, data_paths: Dict[str, str], race_info: Dict[str, Any], 
                                weather_conditions: Dict[str, Any]) -> pd.DataFrame:
        """
        Make race day prediction using the race day model (just before race).
        
        Args:
            data_paths: Paths to data files
            race_info: Information about the race
            weather_conditions: Weather conditions for the race
            
        Returns:
            DataFrame with prediction results
        """
        self.agent_logger.task_start("Making race day prediction")
        
        try:
            # Load historical data
            historical_data = None
            if 'historical_combined' in data_paths:
                historical_data = pd.read_csv(data_paths['historical_combined'])
                self.agent_logger.info(f"Loaded historical data from {data_paths['historical_combined']}")
            
            # Load qualifying data
            qualifying_data = None
            # Look for qualifying data in data_paths
            for key, path in data_paths.items():
                if '_Q' in key:
                    qualifying_data = pd.read_csv(path)
                    self.agent_logger.info(f"Loaded qualifying data from {path}")
                    break
            
            # Load race day specific data (e.g., starting tire compounds, grid penalties)
            race_day_data = None
            # In a real system, this would come from a race day data source
            # For now, we'll use qualifying data as a base and add race day specific features
            if qualifying_data is not None:
                race_day_data = qualifying_data.copy()
                
                # Add race day specific features
                if 'grid_penalties' in race_info:
                    for driver, penalty in race_info['grid_penalties'].items():
                        # Apply grid penalty to driver
                        if driver in race_day_data['Driver'].values:
                            idx = race_day_data[race_day_data['Driver'] == driver].index[0]
                            # Store original grid position
                            race_day_data.loc[idx, 'original_grid_position'] = race_day_data.loc[idx, 'GridPosition']
                            # Apply penalty
                            race_day_data.loc[idx, 'GridPosition'] += penalty
                            race_day_data.loc[idx, 'grid_penalty_applied'] = 1
                
                # Add starting tire compound if available
                if 'starting_tires' in race_info:
                    for driver, compound in race_info['starting_tires'].items():
                        if driver in race_day_data['Driver'].values:
                            idx = race_day_data[race_day_data['Driver'] == driver].index[0]
                            race_day_data.loc[idx, 'starting_tire_compound'] = compound
            
            # If we don't have all necessary data, return empty result
            if historical_data is None or race_day_data is None:
                self.agent_logger.error("Missing required data for race day prediction")
                return pd.DataFrame()
            
            # Clean and prepare data
            cleaned_historical = self.data_cleaner.standardize_driver_names(historical_data)
            cleaned_historical = self.data_cleaner.standardize_team_names(cleaned_historical)
            cleaned_historical = self.data_cleaner.standardize_circuit_names(cleaned_historical, 'TrackName')
            
            cleaned_race_day = self.data_cleaner.standardize_driver_names(race_day_data)
            cleaned_race_day = self.data_cleaner.standardize_team_names(cleaned_race_day)
            cleaned_race_day = self.data_cleaner.standardize_circuit_names(cleaned_race_day, 'TrackName')
            
            # Add grid position features and qualifying features
            race_day_with_features = self.feature_engineer.create_grid_position_features(cleaned_race_day)
            race_day_with_features = self.feature_engineer.create_qualifying_features(race_day_with_features)
            
            # Add circuit and weather information
            race_day_with_features['TrackName'] = race_info.get('circuit', 'unknown')
            race_day_with_features['Year'] = datetime.now().year
            race_day_with_features['Date'] = race_info.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            # Add weather conditions if available
            if weather_conditions:
                for key, value in weather_conditions.items():
                    race_day_with_features[key] = value
            
            # Add track temperature if available
            if 'track_temp_celsius' in race_info:
                race_day_with_features['track_temp_celsius'] = race_info['track_temp_celsius']
            
            # Generate features
            prediction_features = self.feature_engineer.create_all_features(
                race_day_with_features, cleaned_historical, encode_categorical=True
            )
            
            # Make prediction
            if not self.race_day_model.is_trained:
                self.agent_logger.warning("Race day model is not trained. Training on historical data...")
                # Prepare historical data for training
                historical_features = self.feature_engineer.create_all_features(
                    cleaned_historical, encode_categorical=True
                )
                # Train model
                self.race_day_model.train(
                    historical_features, cleaned_historical['Position']
                )
                # Save model
                self.race_day_model.save(os.path.join(self.model_dir, "race_day_model.pkl"))
            
            # Make predictions
            # If other models are trained, use ensemble prediction
            if self.initial_model.is_trained and self.pre_race_model.is_trained:
                predictions = self.race_day_model.ensemble_predict(prediction_features)
            else:
                predictions = self.race_day_model.predict_race_results(prediction_features)
            
            self.agent_logger.task_complete("Making race day prediction")
            return predictions
            
        except Exception as e:
            self.agent_logger.task_fail("Making race day prediction", str(e))
            raise
    
    def _generate_scenario_predictions(self, data_paths: Dict[str, str], race_info: Dict[str, Any], 
                                     weather_conditions: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Generate predictions for different race scenarios.
        
        Args:
            data_paths: Paths to data files
            race_info: Information about the race
            weather_conditions: Weather conditions for the race
            
        Returns:
            Dictionary mapping scenario names to prediction DataFrames
        """
        self.agent_logger.task_start("Generating scenario predictions")
        
        # If race day model isn't trained or loaded, return empty result
        if not self.race_day_model.is_trained:
            self.agent_logger.warning("Race day model is not trained, cannot generate scenario predictions")
            return {}
        
        try:
            # Define scenarios
            scenarios = [
                {
                    'name': 'Dry Race',
                    'description': 'Race with dry conditions',
                    'parameters': {
                        'weather_is_dry': 1,
                        'weather_is_any_wet': 0,
                        'weather_is_very_wet': 0,
                        'racing_condition': 'dry'
                    }
                },
                {
                    'name': 'Wet Race',
                    'description': 'Race with wet conditions',
                    'parameters': {
                        'weather_is_dry': 0,
                        'weather_is_any_wet': 1,
                        'weather_is_very_wet': 0,
                        'racing_condition': 'wet'
                    }
                },
                {
                    'name': 'Very Wet Race',
                    'description': 'Race with heavy rain',
                    'parameters': {
                        'weather_is_dry': 0,
                        'weather_is_any_wet': 1,
                        'weather_is_very_wet': 1,
                        'racing_condition': 'very_wet'
                    }
                },
                {
                    'name': 'Safety Car',
                    'description': 'Race with safety car intervention',
                    'parameters': {
                        'safety_car_deployed': 1
                    }
                },
                {
                    'name': 'Hot Track',
                    'description': 'Higher than expected track temperature',
                    'parameters': {
                        'track_temp_celsius': 45,
                        'weather_temp_hot': 1,
                        'weather_temp_mild': 0
                    }
                }
            ]
            
            # Make race day prediction as base
            base_predictions = self._make_race_day_prediction(data_paths, race_info, weather_conditions)
            
            if base_predictions.empty:
                self.agent_logger.warning("Base predictions are empty, cannot generate scenario predictions")
                return {}
            
            # Generate predictions for each scenario
            scenario_predictions = {}
            
            for scenario in scenarios:
                # Modify weather conditions based on scenario
                scenario_weather = weather_conditions.copy() if weather_conditions else {}
                scenario_weather.update(scenario['parameters'])
                
                # Modify race info based on scenario
                scenario_race_info = race_info.copy()
                for key, value in scenario['parameters'].items():
                    if key not in ['weather_is_dry', 'weather_is_any_wet', 'weather_is_very_wet', 'racing_condition']:
                        scenario_race_info[key] = value
                
                try:
                    # Make prediction for this scenario
                    predictions = self._make_race_day_prediction(data_paths, scenario_race_info, scenario_weather)
                    
                    if not predictions.empty:
                        # Add scenario information
                        predictions['Scenario'] = scenario['name']
                        predictions['ScenarioDescription'] = scenario['description']
                        
                        # Store in results
                        scenario_predictions[scenario['name']] = predictions
                        
                except Exception as e:
                    self.agent_logger.warning(f"Error generating prediction for scenario {scenario['name']}: {str(e)}")
            
            self.agent_logger.task_complete("Generating scenario predictions")
            return scenario_predictions
            
        except Exception as e:
            self.agent_logger.task_fail("Generating scenario predictions", str(e))
            return {}
    
    def _save_prediction(self, predictions: pd.DataFrame, prediction_type: str, 
                       race_info: Dict[str, Any]) -> str:
        """
        Save prediction results to file.
        
        Args:
            predictions: DataFrame with prediction results
            prediction_type: Type of prediction ('initial', 'pre_race', 'race_day')
            race_info: Information about the race
            
        Returns:
            Path to the saved file
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get race name
        race_name = race_info.get('name', 'unknown').replace(' ', '_')
        
        # Create filename
        filename = f"{timestamp}_{prediction_type}_{race_name}"
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        
        # Save data
        try:
            # Save CSV
            predictions.to_csv(csv_path, index=False)
            
            # Also save JSON for easier parsing
            # Convert DataFrame to dict for JSON serialization
            predictions_dict = predictions.to_dict(orient='records')
            
            with open(json_path, 'w') as f:
                json.dump({
                    'prediction_type': prediction_type,
                    'race_info': race_info,
                    'timestamp': timestamp,
                    'predictions': predictions_dict
                }, f, indent=2, default=str)
            
            self.agent_logger.info(f"Saved prediction to {csv_path} and {json_path}")
            
            return csv_path
            
        except Exception as e:
            self.agent_logger.error(f"Error saving prediction to file: {str(e)}")
            raise
    
    def _get_predictions_summary(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the predictions for notification purposes.
        
        Args:
            predictions: DataFrame with prediction results
            
        Returns:
            Dictionary with prediction summary
        """
        if predictions is None or predictions.empty:
            return {'error': 'No predictions available'}
        
        try:
            # Get top 3 predicted drivers
            top3 = predictions.head(3)
            
            summary = {
                'podium': top3['Driver'].tolist(),
                'winner': top3.iloc[0]['Driver'] if len(top3) > 0 else None,
                'total_drivers': len(predictions)
            }
            
            return summary
            
        except Exception as e:
            self.agent_logger.error(f"Error creating prediction summary: {str(e)}")
            return {'error': str(e)}