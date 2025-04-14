"""
Tests for the initial model of the F1 prediction project.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from models.initial_model import F1InitialModel

class TestInitialModel(unittest.TestCase):
    """Test cases for the F1InitialModel class."""
    
    def setUp(self):
        """Set up test data and model instance."""
        # Create model instance
        self.model = F1InitialModel(estimator='rf', target='Position')
        
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
            
            # Circuit features
            'circuit_high_speed': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'circuit_street': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'circuit_technical': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            # Weather features
            'weather_is_dry': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            'weather_is_any_wet': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            'weather_temp_mild': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'weather_temp_hot': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'weather_high_wind': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            # Create some interaction features
            'team_highspeed_interaction': [20, 15, 16, 16, 20, 10, 8, 15, 20, 15],
            'wet_driver_advantage_interaction': [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1.0]
        })
        
        # Create sample new race data (different race, same drivers)
        self.new_race_data = pd.DataFrame({
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
            
            # Circuit features
            'circuit_high_speed': [1, 1, 1, 1, 1],
            'circuit_street': [0, 0, 0, 0, 0],
            'circuit_technical': [0, 0, 0, 0, 0],
            
            # Weather features - forecast for the race weekend
            'weather_is_dry': [1, 1, 1, 1, 1],
            'weather_is_any_wet': [0, 0, 0, 0, 0],
            'weather_temp_mild': [1, 1, 1, 1, 1],
            'weather_temp_hot': [0, 0, 0, 0, 0],
            'weather_high_wind': [1, 1, 1, 1, 1],
            
            # Interaction features
            'team_highspeed_interaction': [20, 15, 16, 16, 20],
            'wet_driver_advantage_interaction': [0, 0, 0, 0, 0]
        })
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, 'initial_model')
        self.assertEqual(self.model.estimator_type, 'rf')
        self.assertEqual(self.model.target, 'Position')
        self.assertFalse(self.model.is_trained)
        self.assertIsNotNone(self.model.features)
        self.assertGreater(len(self.model.features), 0)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        # Prepare features from historical data
        features = self.model.prepare_features(self.historical_data)
        
        # Check that features DataFrame has expected shape
        self.assertGreater(len(features), 0)
        self.assertGreater(len(features.columns), 0)
        
        # Check that only relevant features were selected
        for col in features.columns:
            if col not in ['Driver', 'TrackName', 'Circuit']:  # Reference columns are allowed
                self.assertIn(col, self.model.features)
    
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
        
        # Check internal state
        self.assertEqual(self.model.target, 'Position')
        self.assertGreater(len(self.model.features), 0)
    
    def test_predict(self):
        """Test model prediction."""
        # Prepare training data
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        
        # Train the model
        self.model.train(X_train, y_train)
        
        # Prepare test data
        X_test = self.model.prepare_features(self.new_race_data)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.new_race_data))
        self.assertTrue(np.all(predictions >= 1))  # All positions should be at least 1
        self.assertTrue(np.all(predictions <= len(self.new_race_data)))  # No position beyond number of drivers
    
    def test_predict_race_results(self):
        """Test prediction of complete race results."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Predict race results
        results = self.model.predict_race_results(self.new_race_data)
        
        # Check result format
        self.assertEqual(len(results), len(self.new_race_data))
        self.assertIn('Driver', results.columns)
        self.assertIn('PredictedPosition', results.columns)
        
        # Check that positions are unique and valid
        positions = results['PredictedPosition'].values
        self.assertEqual(len(positions), len(set(positions)))  # Unique positions
        self.assertTrue(np.all(positions >= 1))  # All positions should be at least 1
        self.assertTrue(np.all(positions <= len(self.new_race_data)))  # No position beyond number of drivers
        
        # Check that the drivers are preserved
        for driver in self.new_race_data['Driver']:
            self.assertIn(driver, results['Driver'].values)
    
    def test_predict_top_n(self):
        """Test prediction of top N drivers."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Predict top 3
        top3 = self.model.predict_top_n(self.new_race_data, n=3)
        
        # Check result
        self.assertEqual(len(top3), 3)
        self.assertTrue(np.all(top3['PredictedPosition'].values <= 3))
    
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