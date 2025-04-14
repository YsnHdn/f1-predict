"""
Integration tests for the prediction models of the F1 prediction project.
Tests the progressive prediction workflow from initial model to race day model.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.initial_model import F1InitialModel
from models.pre_race_model import F1PreRaceModel
from models.race_day_model import F1RaceDayModel

class TestModelsIntegration(unittest.TestCase):
    """Integration tests for the F1 prediction models."""
    
    def setUp(self):
        """Set up test data and model instances."""
        # Create all model instances
        self.initial_model = F1InitialModel(estimator='rf', target='Position')
        self.pre_race_model = F1PreRaceModel(estimator='rf', target='Position')
        self.race_day_model = F1RaceDayModel(estimator='rf', target='Position')
        
        # Link models for blending and ensemble predictions
        self.pre_race_model.initial_model = self.initial_model
        self.race_day_model.initial_model = self.initial_model
        self.race_day_model.pre_race_model = self.pre_race_model
        
        # Create sample historical data for training
        self.historical_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER', 'NOR', 'ALO', 'RUS'] * 3,
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull', 
                    'McLaren', 'Aston Martin', 'Mercedes'] * 3,
            'Position': [1, 3, 2, 4, 5, 6, 7, 8, 2, 1, 3, 5, 4, 6, 7, 8, 1, 2, 3, 4, 6, 5, 8, 7],
            'Points': [25, 15, 18, 12, 10, 8, 6, 4, 18, 25, 15, 10, 12, 8, 6, 4, 25, 18, 15, 12, 8, 10, 4, 6],
            'GridPosition': [1, 3, 2, 4, 5, 6, 7, 8, 1, 2, 3, 5, 4, 6, 7, 8, 1, 3, 2, 4, 5, 6, 8, 7],
            'Status': ['Finished'] * 24,
            'TrackName': ['monza', 'monza', 'monza', 'monza', 'monza', 'monza', 'monza', 'monza',
                         'spa', 'spa', 'spa', 'spa', 'spa', 'spa', 'spa', 'spa',
                         'silverstone', 'silverstone', 'silverstone', 'silverstone', 
                         'silverstone', 'silverstone', 'silverstone', 'silverstone'],
            'Year': [2022] * 24,
            'Date': [datetime(2022, 9, 11)] * 8 + [datetime(2022, 8, 28)] * 8 + [datetime(2022, 7, 3)] * 8,
            
            # Add minimal subset of features needed for all models
            'driver_avg_points': [23, 20, 16, 14, 12, 10, 8, 6] * 3,
            'driver_finish_rate': [0.95, 0.9, 0.92, 0.88, 0.85, 0.93, 0.91, 0.89] * 3,
            'team_avg_points': [20, 15, 16, 16, 20, 10, 8, 15] * 3,
            'circuit_high_speed': [1] * 24,
            'circuit_street': [0] * 24,
            'weather_is_dry': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            'weather_is_any_wet': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            
            # Qualifying data
            'Q1_seconds': [80.123, 80.456, 80.234, 80.567, 80.789, 81.012, 81.234, 80.901] * 3,
            'Q2_seconds': [79.123, 79.456, 79.234, 79.567, 79.789, 80.012, np.nan, np.nan] * 3,
            'Q3_seconds': [78.123, 78.456, 78.234, 78.567, 78.789, np.nan, np.nan, np.nan] * 3,
            
            # Race day features
            'starting_tire_compound': ['soft', 'soft', 'medium', 'medium', 'soft', 'soft', 'soft', 'medium'] * 3,
            'track_temp_celsius': [35] * 8 + [25] * 8 + [30] * 8
        })
        
        # Create sample new grand prix data for predictions - before qualifying
        race_date = datetime(2023, 7, 16)
        
        self.pre_race_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull'],
            'TrackName': ['silverstone'] * 5,
            'Date': [race_date] * 5,
            
            # Basic historical features
            'driver_avg_points': [23, 20, 16, 14, 12],
            'driver_finish_rate': [0.95, 0.9, 0.92, 0.88, 0.85],
            'team_avg_points': [20, 15, 16, 16, 20],
            'circuit_high_speed': [1] * 5,
            'circuit_street': [0] * 5,
            'weather_is_dry': [1] * 5,
            'weather_is_any_wet': [0] * 5
        })
        
        # Add qualifying data 
        self.qualifying_data = self.pre_race_data.copy()
        self.qualifying_data['GridPosition'] = [1, 3, 2, 4, 5]
        self.qualifying_data['Q1_seconds'] = [79.123, 79.556, 79.234, 79.867, 79.989]
        self.qualifying_data['Q2_seconds'] = [78.123, 78.556, 78.234, 78.867, 78.989]
        self.qualifying_data['Q3_seconds'] = [77.123, 77.556, 77.234, 77.867, 77.989]
        self.qualifying_data['q_sessions_completed'] = [3] * 5
        
        # Add grid position features
        self.qualifying_data['grid_front_row'] = [1, 0, 1, 0, 0]
        self.qualifying_data['grid_top3'] = [1, 1, 1, 0, 0]
        self.qualifying_data['grid_top5'] = [1, 1, 1, 1, 1]
        
        # Add race day data with last-minute information
        self.race_day_data = self.qualifying_data.copy()
        
        # Change weather forecast - now expecting rain
        self.race_day_data['weather_is_dry'] = [0] * 5
        self.race_day_data['weather_is_any_wet'] = [1] * 5 
        self.race_day_data['RaceWeatherCondition'] = ['light rain'] * 5
        
        # Add race day specific features
        self.race_day_data['GridPenalty'] = [0, 0, 0, 0, 1]  # PER got a penalty
        self.race_day_data['original_grid_position'] = self.race_day_data['GridPosition'].copy()
        self.race_day_data.loc[self.race_day_data['GridPenalty'] == 1, 'GridPosition'] += 1
        self.race_day_data['StartingTireCompound'] = ['intermediate'] * 5
        self.race_day_data['track_temp_celsius'] = [22] * 5
    
    def test_progressive_prediction_workflow(self):
        """Test the entire prediction workflow from initial to race day model."""
        # Step 1: Train all models on historical data
        # First, handle the feature preparation for each model
        X_train_initial = self.initial_model.prepare_features(self.historical_data)
        X_train_prerace = self.pre_race_model.prepare_features(self.historical_data)
        X_train_raceday = self.race_day_model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        
        # Train each model
        self.initial_model.train(X_train_initial, y_train)
        self.pre_race_model.train(X_train_prerace, y_train)
        self.race_day_model.train(X_train_raceday, y_train)
        
        # Verify that all models are trained
        self.assertTrue(self.initial_model.is_trained)
        self.assertTrue(self.pre_race_model.is_trained)
        self.assertTrue(self.race_day_model.is_trained)
        
        # Step 2: Make initial predictions before qualifying
        initial_predictions = self.initial_model.predict_race_results(self.pre_race_data)
        
        # Verify initial predictions
        self.assertEqual(len(initial_predictions), len(self.pre_race_data))
        self.assertIn('Driver', initial_predictions.columns)
        self.assertIn('PredictedPosition', initial_predictions.columns)
        
        # Step 3: Make pre-race predictions after qualifying
        pre_race_predictions = self.pre_race_model.predict_race_results(self.qualifying_data)
        
        # Also test blending with initial predictions
        blended_predictions = self.pre_race_model.blend_with_initial_model(self.qualifying_data)
        
        # Verify pre-race predictions
        self.assertEqual(len(pre_race_predictions), len(self.qualifying_data))
        self.assertEqual(len(blended_predictions), len(self.qualifying_data))
        
        # Step 4: Make race day predictions with last-minute information
        race_day_predictions = self.race_day_model.predict_race_results(self.race_day_data)
        
        # Also test ensemble predictions
        ensemble_predictions = self.race_day_model.ensemble_predict(self.race_day_data)
        
        # Verify race day predictions
        self.assertEqual(len(race_day_predictions), len(self.race_day_data))
        self.assertEqual(len(ensemble_predictions), len(self.race_day_data))
        
        # Step 5: Compare predictions across models to ensure they evolve
        # Extract predictions for the same driver across models
        driver = 'VER'
        initial_pos = initial_predictions.loc[initial_predictions['Driver'] == driver, 'PredictedPosition'].values[0]
        pre_race_pos = pre_race_predictions.loc[pre_race_predictions['Driver'] == driver, 'PredictedPosition'].values[0]
        race_day_pos = race_day_predictions.loc[race_day_predictions['Driver'] == driver, 'PredictedPosition'].values[0]
        
        # Log the evolution of predictions
        print(f"\nPrediction evolution for {driver}:")
        print(f"Initial model: Position {initial_pos}")
        print(f"Pre-race model: Position {pre_race_pos}")
        print(f"Race day model: Position {race_day_pos}")
        
        # We don't assert equality or inequality because predictions could legitimately be the same,
        # but they should all be valid positions
        self.assertTrue(1 <= initial_pos <= len(self.pre_race_data))
        self.assertTrue(1 <= pre_race_pos <= len(self.qualifying_data))
        self.assertTrue(1 <= race_day_pos <= len(self.race_day_data))
    
    def test_scenario_predictions(self):
        """Test scenario-based predictions with the race day model."""
        # Train the race day model
        X_train = self.race_day_model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.race_day_model.train(X_train, y_train)
        
        # Define scenarios
        scenarios = [
            {
                'name': 'Base Scenario',
                'description': 'Current conditions',
                'parameters': {}
            },
            {
                'name': 'Dry Race',
                'description': 'Race with dry conditions',
                'parameters': {
                    'weather_is_dry': 1,
                    'weather_is_any_wet': 0
                }
            },
            {
                'name': 'Very Wet Race',
                'description': 'Race with heavy rain',
                'parameters': {
                    'weather_is_dry': 0,
                    'weather_is_any_wet': 1,
                    'weather_racing_condition': 'very_wet'
                }
            }
        ]
        
        # Generate scenario predictions
        scenario_predictions = self.race_day_model.predict_with_scenarios(self.race_day_data, scenarios)
        
        # Verify scenario predictions
        self.assertEqual(len(scenario_predictions), len(scenarios))
        
        # Check that each scenario has predictions for all drivers
        for scenario_name, predictions in scenario_predictions.items():
            self.assertEqual(len(predictions), len(self.race_day_data))
            self.assertIn('Scenario', predictions.columns)
            self.assertEqual(predictions['Scenario'].iloc[0], scenario_name)
        
        # Compare predictions across scenarios
        # We expect different conditions to potentially affect driver performance
        base_scenario = scenario_predictions['Base Scenario']
        dry_scenario = scenario_predictions['Dry Race']
        wet_scenario = scenario_predictions['Very Wet Race']
        
        # Focus on a driver with good wet performance
        driver = 'HAM'  # Historically good in wet conditions
        
        base_pos = base_scenario.loc[base_scenario['Driver'] == driver, 'PredictedPosition'].values[0]
        dry_pos = dry_scenario.loc[dry_scenario['Driver'] == driver, 'PredictedPosition'].values[0]
        wet_pos = wet_scenario.loc[wet_scenario['Driver'] == driver, 'PredictedPosition'].values[0]
        
        print(f"\nScenario predictions for {driver}:")
        print(f"Base scenario: Position {base_pos}")
        print(f"Dry race: Position {dry_pos}")
        print(f"Very wet race: Position {wet_pos}")
        
        # Again, we don't assert specific relationships as they depend on the model's learned patterns


if __name__ == '__main__':
    unittest.main()