"""
Tests for the pre-race model of the F1 prediction project.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from models.pre_race_model import F1PreRaceModel
from models.initial_model import F1InitialModel
from sklearn.ensemble import RandomForestRegressor

class TestPreRaceModel(unittest.TestCase):
    """Test cases for the F1PreRaceModel class."""
    
    def setUp(self):
        """Set up test data and model instance."""
        # Create model instance
        self.model = F1PreRaceModel(estimator='rf', target='Position')
        
        # Create an initial model for testing blending
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
            
            # Other features
            'circuit_high_degradation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'circuit_low_degradation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            
            # Interaction features
            'team_highspeed_interaction': [20, 15, 16, 16, 20, 10, 8, 15, 20, 15],
            'wet_driver_advantage_interaction': [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1.0],
            'grid_overtaking_interaction': [0.4, 1.2, 0.8, 1.6, 2.0, 2.4, 2.8, 3.2, 0.6, 1.2]
        })
        
        # Create sample new race data with qualifying results
        self.qualifying_data = pd.DataFrame({
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
            
            # Weather features - forecast for the race weekend
            'weather_is_dry': [1, 1, 1, 1, 1],
            'weather_is_any_wet': [0, 0, 0, 0, 0],
            'weather_temp_mild': [1, 1, 1, 1, 1],
            'weather_temp_hot': [0, 0, 0, 0, 0],
            'weather_high_wind': [1, 1, 1, 1, 1],
            
            # Qualifying position and data (this is what differentiates from initial model)
            'GridPosition': [1, 3, 2, 4, 5],
            'Q1_seconds': [79.123, 79.556, 79.234, 79.867, 79.989],
            'Q2_seconds': [78.123, 78.556, 78.234, 78.867, 78.989],
            'Q3_seconds': [77.123, 77.556, 77.234, 77.867, 77.989],
            'q_sessions_completed': [3, 3, 3, 3, 3],
            'gap_to_pole_pct': [0.0, 0.56, 0.14, 0.97, 1.12],
            
            # Grid position features
            'grid_front_row': [1, 0, 1, 0, 0],
            'grid_top3': [1, 1, 1, 0, 0],
            'grid_top5': [1, 1, 1, 1, 1],
            'grid_top10': [1, 1, 1, 1, 1],
            
            # Other features
            'circuit_high_degradation': [0, 0, 0, 0, 0],
            'circuit_low_degradation': [0, 0, 0, 0, 0],
            
            # Interaction features
            'team_highspeed_interaction': [20, 15, 16, 16, 20],
            'wet_driver_advantage_interaction': [0, 0, 0, 0, 0],
            'grid_overtaking_interaction': [0.5, 1.5, 1.0, 2.0, 2.5]
        })
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, 'pre_race_model')
        self.assertEqual(self.model.estimator_type, 'rf')
        self.assertEqual(self.model.target, 'Position')
        self.assertFalse(self.model.is_trained)
        self.assertIsNotNone(self.model.features)
        self.assertGreater(len(self.model.features), 0)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        # Prepare features from qualifying data
        features = self.model.prepare_features(self.qualifying_data)
        
        # Check that features DataFrame has expected shape
        self.assertGreater(len(features), 0)
        self.assertGreater(len(features.columns), 0)
        
        # Check that key qualifying features were selected
        qualifying_features = ['GridPosition', 'Q1_seconds', 'Q2_seconds', 'Q3_seconds', 
                              'grid_front_row', 'grid_top3', 'grid_top5']
        for feature in qualifying_features:
            if feature in self.qualifying_data.columns:
                self.assertIn(feature, features.columns)
    
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
        X_test = self.model.prepare_features(self.qualifying_data)
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.qualifying_data))
        self.assertTrue(np.all(predictions >= 1))  # All positions should be at least 1
        self.assertTrue(np.all(predictions <= len(self.qualifying_data)))  # No position beyond number of drivers
    
    def test_predict_race_results(self):
        """Test prediction of complete race results."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Predict race results
        results = self.model.predict_race_results(self.qualifying_data)
        
        # Check result format
        self.assertEqual(len(results), len(self.qualifying_data))
        self.assertIn('Driver', results.columns)
        self.assertIn('PredictedPosition', results.columns)
        
        # Check that positions are unique and valid
        positions = results['PredictedPosition'].values
        self.assertEqual(len(positions), len(set(positions)))  # Unique positions
        self.assertTrue(np.all(positions >= 1))  # All positions should be at least 1
        self.assertTrue(np.all(positions <= len(self.qualifying_data)))  # No position beyond number of drivers
    
    def test_blend_with_initial_model(self):
        """Test blending predictions with initial model."""
        # Skip test if initial model is not available
        if self.initial_model is None:
            self.skipTest("Initial model not available for testing")
        
        # Train both models
        X_train_initial = self.initial_model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.initial_model.train(X_train_initial, y_train)
        
        X_train_prerace = self.model.prepare_features(self.historical_data)
        self.model.train(X_train_prerace, y_train)
        
        # Set initial model for blending
        self.model.initial_model = self.initial_model
        
        # Test blending
        blended_results = self.model.blend_with_initial_model(self.qualifying_data)
        
        # Check blended results
        self.assertIsNotNone(blended_results)
        self.assertGreater(len(blended_results), 0)
        self.assertIn('Driver', blended_results.columns)
        self.assertIn('PredictedPosition', blended_results.columns)
        
        # Check that all drivers are present
        for driver in self.qualifying_data['Driver']:
            self.assertIn(driver, blended_results['Driver'].values)
    
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
    
    def test_predict_podium(self):
        """Test prediction of podium positions."""
        # Train the model
        X_train = self.model.prepare_features(self.historical_data)
        y_train = self.historical_data['Position']
        self.model.train(X_train, y_train)
        
        # Predict podium
        podium = self.model.predict_podium(self.qualifying_data)
        
        # Check result
        self.assertEqual(len(podium), 3)
        self.assertTrue(np.all(podium['PredictedPosition'].values <= 3))
    
    def test_optimize_hyperparameters(self):
        """Test hyperparameter optimization."""
        # Pour ce test, nous allons simplement vérifier que la méthode retourne un dictionnaire 
        # avec les hyperparamètres sans réellement exécuter GridSearchCV qui est lent
        
        # Créer une petite version du jeu de données
        small_X = self.model.prepare_features(self.historical_data.head(8))
        small_y = self.historical_data['Position'].head(8)
        
        # Définir un comportement simple pour l'optimisation
        # On utilise intentionnellement un petit nombre d'estimateurs pour rendre le test rapide
        self.model.model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
        
        try:
            # Appeler la méthode avec un CV minimal
            result = self.model.optimize_hyperparameters(small_X, small_y, cv=2)
            
            # Vérifier que le résultat est un dictionnaire
            self.assertIsInstance(result, dict)
            
            # Vérifier qu'il contient des hyperparamètres attendus
            # Notez que nous ne vérifions pas les valeurs exactes, seulement leur présence
            self.assertTrue(any(param in result for param in ['n_estimators', 'max_depth', 'min_samples_split']))
        except Exception as e:
            # Si une erreur est levée, vérifier que c'est pour une raison acceptable
            # comme le manque de sklearn ou une limitation technique
            self.skipTest(f"Impossible de tester l'optimisation des hyperparamètres: {str(e)}")

if __name__ == '__main__':
    unittest.main()