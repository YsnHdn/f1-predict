"""
Tests for the race day model of the F1 prediction project.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from models.race_day_model import F1RaceDayModel
from models.pre_race_model import F1PreRaceModel
from models.initial_model import F1InitialModel

class TestRaceDayModel(unittest.TestCase):
    """Test cases for the F1RaceDayModel class."""
    
    def setUp(self):
        """Set up test data and model instance."""
        # Create model instances
        self.model = F1RaceDayModel(estimator='rf', target='Position')
        self.pre_race_model = F1PreRaceModel(estimator='rf', target='Position')
        self.initial_model = F1InitialModel(estimator='rf', target='Position')
        
        # Create sample historical data with relevant features
        self.historical_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER', 'NOR', 'ALO', 'RUS', 'VER', 'HAM'],
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull', 
                    'McLaren', 'Aston Martin', 'Mercedes', 'Red Bull', 'Mercedes'],
            'Position': [1, 3, 2, 4, 5, 6, 7, 8, 2, 1],
            'Points': [25, 15, 18, 12, 10, 8, 6, 4, 18, 25],
            'GridPosition': [1, 3, 2, 4, 5, 6, 7, 8, 1, 2],
            'Status': ['Finished', 'Finished', 'Finished', 'Finished', 'Finished',
                      'Finished', 'Finished', 'Finished', 'Finished', 'Finished'],
            'TrackName': ['monza', 'monza', 'monza', 'monza', 'monza',
                         'monza', 'monza', 'monza', 'spa', 'spa'],
            'Year': [2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
            'Date': [datetime(2022, 9, 11), datetime(2022, 9, 11), datetime(2022, 9, 11),
                    datetime(2022, 9, 11), datetime(2022, 9, 11), datetime(2022, 9, 11),
                    datetime(2022, 9, 11), datetime(2022, 9, 11), datetime(2022, 8, 28),
                    datetime(2022, 8, 28)],
            
            # Driver features
            'driver_avg_points': [23, 20, 16, 14, 12, 10, 8, 6, 23, 20],
            'driver_avg_positions_gained': [0.5, 1.0, 0.2, 0.3, 0.1, 0.4, 0.6, 0.7, 0.5, 1.0],
            'driver_finish_rate': [0.95, 0.90, 0.92, 0.88, 0.85, 0.93, 0.94, 0.91, 0.95, 0.90],
            'driver_form_trend': [-0.2, 0.3, 0.1, -0.1, 0.2, 0.4, -0.3, 0.5, -0.2, 0.3],
            
            # Team features
            'team_avg_points': [20, 15, 16, 16, 20, 10, 8, 15, 20, 15],
            'team_finish_rate': [0.92, 0.9, 0.88, 0.88, 0.92, 0.85, 0.8, 0.9, 0.92, 0.9],
            'grid_vs_teammate': [-2.0, 0, -0.5, 0.5, 2.0, 0, 0, 0, -0.5, 0.5],
            
            # Circuit features
            'circuit_high_speed': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'circuit_street': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'circuit_technical': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'overtaking_difficulty': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6],
            
            # Weather features
            'weather_is_dry': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            'weather_is_any_wet': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            'weather_temp_mild': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'weather_temp_hot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'weather_high_wind': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            # Qualifying data
            'Q1_seconds': [80.123, 80.456, 80.234, 80.567, 80.789, 81.012, 81.234, 80.901, 93.123, 93.234],
            'Q2_seconds': [79.123, 79.456, 79.234, 79.567, 79.789, 80.012, np.nan, np.nan, 92.123, 92.234],
            'Q3_seconds': [78.123, 78.456, 78.234, 78.567, 78.789, np.nan, np.nan, np.nan, 91.123, 91.234],
            'q_sessions_completed': [3, 3, 3, 3, 3, 2, 1, 1, 3, 3],
            'gap_to_pole_pct': [0.0, 0.4, 0.2, 0.6, 0.8, 1.2, np.nan, np.nan, 0.0, 0.1],
            
            # Grid position features
            'grid_front_row': [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
            'grid_top3': [1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
            'grid_top5': [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
            'grid_top10': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            
            # Race day features
            'starting_tire_compound': ['soft', 'soft', 'medium', 'medium', 'soft', 
                                      'soft', 'soft', 'medium', 'wet', 'wet'],
            'grid_penalty_applied': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'original_grid_position': [1, 3, 2, 4, 5, 6, 7, 8, 1, 2],
            'formation_lap_issue': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'track_temp_celsius': [35, 35, 35, 35, 35, 35, 35, 35, 25, 25],
            'track_evolution': [7, 7, 7, 7, 7, 7, 7, 7, 5, 5],
            'pit_lane_position': [1, 5, 3, 4, 2, 6, 9, 8, 1, 5],
            
            # Other features
            'circuit_high_degradation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'circuit_low_degradation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            # Interaction features
            'team_highspeed_interaction': [20, 15, 16, 16, 20, 10, 8, 15, 20, 15],
            'wet_driver_advantage_interaction': [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1.0],
            'grid_overtaking_interaction': [0.4, 1.2, 0.8, 1.6, 2.0, 2.4, 2.8, 3.2, 0.6, 1.2]
        })
        
        # Create sample race day data with last-minute information
        self.race_day_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull'],
            'TrackName': ['silverstone', 'silverstone', 'silverstone', 'silverstone', 'silverstone'],
            'Year': [2023, 2023, 2023, 2023, 2023],
            'Date': [datetime(2023, 7, 16), datetime(2023, 7, 16), datetime(2023, 7, 16),
                    datetime(2023, 7, 16), datetime(2023, 7, 16)],
            
            # Driver features (copied from historical)
            'driver_avg_points': [23, 20, 16, 14, 12],
            'driver_avg_positions_gained': [0.5, 1.0, 0.2, 0.3, 0.1],
            'driver_finish_rate': [0.95, 0.90, 0.92, 0.88, 0.85],
            'driver_form_trend': [-0.2, 0.3, 0.1, -0.1, 0.2],
            
            # Team features
            'team_avg_points': [20, 15, 16, 16, 20],
            'team_finish_rate': [0.92, 0.9, 0.88, 0.88, 0.92],
            'grid_vs_teammate': [-2.0, 0, -0.5, 0.5, 2.0],
            
            # Circuit features
            'circuit_high_speed': [1, 1, 1, 1, 1],
            'circuit_street': [0, 0, 0, 0, 0],
            'circuit_technical': [0, 0, 0, 0, 0],
            'overtaking_difficulty': [0.5, 0.5, 0.5, 0.5, 0.5],
            
            # Weather features - updated on race day
            'weather_is_dry': [0, 0, 0, 0, 0],  # Now expecting rain
            'weather_is_any_wet': [1, 1, 1, 1, 1],
            'weather_temp_mild': [1, 1, 1, 1, 1],
            'weather_temp_hot': [0, 0, 0, 0, 0],
            'weather_high_wind': [1, 1, 1, 1, 1],
            'RaceWeatherCondition': ['light rain', 'light rain', 'light rain', 'light rain', 'light rain'],
            
            # Qualifying position and data 
            'GridPosition': [1, 3, 2, 4, 6],  # PER got a penalty
            'Q1_seconds': [79.123, 79.556, 79.234, 79.867, 79.989],
            'Q2_seconds': [78.123, 78.556, 78.234, 78.867, 78.989],
            'Q3_seconds': [77.123, 77.556, 77.234, 77.867, 77.989],
            'q_sessions_completed': [3, 3, 3, 3, 3],
            'gap_to_pole_pct': [0.0, 0.56, 0.14, 0.97, 1.12],
            
            # Grid position features
            'grid_front_row': [1, 0, 1, 0, 0],
            'grid_top3': [1, 1, 1, 0, 0],
            'grid_top5': [1, 1, 1, 1, 0],
            'grid_top10': [1, 1, 1, 1, 1],
            
            # Race day specific features
            'GridPenalty': [0, 0, 0, 0, 1],  # PER got a penalty
            'StartingTireCompound': ['intermediate', 'intermediate', 'intermediate', 
                                   'intermediate', 'intermediate'],
            'original_grid_position': [1, 3, 2, 4, 5],
            'formation_lap_issue': [0, 0, 0, 0, 0],
            'track_temp_celsius': [22, 22, 22, 22, 22],  # Cooler due to rain
            'track_evolution': [4, 4, 4, 4, 4],  # Lower due to rain
            'pit_lane_position': [1, 5, 3, 4, 2],
            'team_recent_strategy': ['2stop', '2stop', '2stop', '2stop', '2stop'],
            'recent_race_pace': [77.5, 77.8, 77.6, 77.9, 77.7],
            'car_upgrades_applied': [1, 0, 1, 1, 0],
            
            # Driver wet performance
            'driver_wet_advantage': [0.2, 1.5, -0.5, 0.1, 0.3],
            
            # Other features
            'circuit_high_degradation': [0, 0, 0, 0, 0],
            'circuit_low_degradation': [0, 0, 0, 0, 0],
            
            # Interaction features
            'team_highspeed_interaction': [20, 15, 16, 16, 20],
            'wet_driver_advantage_interaction': [0, 0, 0, 0, 0],
            'grid_overtaking_interaction': [0.5, 1.5, 1.0, 2.0, 3.0]
        })
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, 'race_day_model')
        self.assertEqual(self.model.estimator_type, 'rf')
        self.assertEqual(self.model.target, 'Position')
        self.assertFalse(self.model.is_trained)
        self.assertIsNotNone(self.model.features)
        self.assertGreater(len(self.model.features), 0)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        # Prepare features from race day data
        features = self.model.prepare_features(self.race_day_data)
        
        # Check that features DataFrame has expected shape
        self.assertGreater(len(features), 0)
        self.assertGreater(len(features.columns), 0)
        
        # Check that key race day features were included
        race_day_features = ['grid_penalty_applied', 'track_temp_celsius', 'starting_tire_compound', 
                           'original_grid_position']
        
        # Some features might be transformed or renamed
        for feature in race_day_features:
            found = False
            for col in features.columns:
                if feature in col:
                    found = True
                    break
            if not found and feature in self.race_day_data.columns:
                self.fail(f"Race day feature '{feature}' not found in prepared features")
    
    def test_train(self):
        """Test model training."""
        # Prepare features
        X = self.model.prepare_features(self.historical_data)
        y = self.historical_data['Position']
        
        # Train the model
        self.model.train(X, y)
        
        # Check that model is trained
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.training_date)
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        """Test model prediction."""
        # Prepare training data
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        
        # Train the model
        self.model.train(X_train, y_train)
        
        # Prepare test data
        X_test = self.model.prepare_features(self.race_day_data)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.race_day_data))
        self.assertTrue(np.all(predictions >= 1))  # All positions should be at least 1
        self.assertTrue(np.all(predictions <= len(self.race_day_data)))  # No position beyond number of drivers
    
    def test_predict_race_results(self):
        """Test prediction of complete race results."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Predict race results
        results = self.model.predict_race_results(self.race_day_data)
        
        # Check result format
        self.assertEqual(len(results), len(self.race_day_data))
        self.assertIn('Driver', results.columns)
        self.assertIn('PredictedPosition', results.columns)
        
        # Check that positions are unique and valid
        positions = results['PredictedPosition'].values
        self.assertEqual(len(positions), len(set(positions)))  # Unique positions
        self.assertTrue(np.all(positions >= 1))  # All positions should be at least 1
        self.assertTrue(np.all(positions <= len(self.race_day_data)))  # No position beyond number of drivers
    
    def test_ensemble_predict(self):
        """Test ensemble prediction with multiple models."""
        # Skip if models are not available
        if self.pre_race_model is None or self.initial_model is None:
            self.skipTest("Required models not available for testing")
        
        # Train all models
        X_train_initial = self.initial_model.prepare_features(self.historical_data)
        X_train_prerace = self.pre_race_model.prepare_features(self.historical_data)
        X_train_raceday = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        
        self.initial_model.train(X_train_initial, y_train)
        self.pre_race_model.train(X_train_prerace, y_train)
        self.model.train(X_train_raceday, y_train)
        
        # Set models for ensemble prediction
        self.model.initial_model = self.initial_model
        self.model.pre_race_model = self.pre_race_model
        
        # Test ensemble prediction
        ensemble_results = self.model.ensemble_predict(
            self.race_day_data,
            weights={'race_day': 0.6, 'pre_race': 0.3, 'initial': 0.1}
        )
        
        # Check ensemble results
        self.assertIsNotNone(ensemble_results)
        self.assertGreater(len(ensemble_results), 0)
        self.assertIn('Driver', ensemble_results.columns)
        self.assertIn('PredictedPosition', ensemble_results.columns)
        
        # Check that all drivers are present with valid positions
        for driver in self.race_day_data['Driver']:
            driver_row = ensemble_results[ensemble_results['Driver'] == driver]
            self.assertEqual(len(driver_row), 1)
            position = driver_row['PredictedPosition'].values[0]
            self.assertTrue(1 <= position <= len(self.race_day_data))
    
    def test_predict_with_scenarios(self):
        """Test prediction under different scenarios."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Define test scenarios
        scenarios = [
            {
                'name': 'Dry Conditions',
                'description': 'Race with dry weather',
                'parameters': {
                    'weather_is_dry': 1,
                    'weather_is_any_wet': 0
                }
            },
            {
                'name': 'Heavy Rain',
                'description': 'Race with heavy rain',
                'parameters': {
                    'weather_is_dry': 0,
                    'weather_is_any_wet': 1,
                    'weather_racing_condition': 'very_wet'
                }
            }
        ]
        
        # Get scenario predictions
        scenario_predictions = self.model.predict_with_scenarios(self.race_day_data, scenarios)
        
        # Check scenario predictions
        self.assertEqual(len(scenario_predictions), len(scenarios))
        
        for scenario_name, predictions in scenario_predictions.items():
            # Each scenario should have predictions for all drivers
            self.assertEqual(len(predictions), len(self.race_day_data))
            self.assertIn('Driver', predictions.columns)
            self.assertIn('PredictedPosition', predictions.columns)
            self.assertIn('Scenario', predictions.columns)
            
            # Check that scenario name is correctly set
            self.assertEqual(predictions['Scenario'].iloc[0], scenario_name)
    
    def test_get_feature_importance(self):
        """Test retrieval of feature importance."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Check that importance is available
        self.assertIsNotNone(importance)
        if importance is not None:
            self.assertIn('Feature', importance.columns)
            self.assertIn('Importance', importance.columns)
            self.assertGreater(len(importance), 0)


if __name__ == '__main__':
    unittest.main()