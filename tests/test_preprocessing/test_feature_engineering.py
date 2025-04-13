"""
Tests for the feature engineering module of the F1 prediction project.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from preprocessing.feature_engineering import F1FeatureEngineer

class TestF1FeatureEngineer(unittest.TestCase):
    """Test cases for the F1FeatureEngineer class."""
    
    def setUp(self):
        """Set up test data and feature engineer instance."""
        self.engineer = F1FeatureEngineer(scale_features=True, scaling_method='standard')
        
        # Create sample race data with qualifying and weather
        self.race_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull'],
            'Position': [1, 2, 3, 4, 5],
            'Points': [25, 18, 15, 12, 10],
            'GridPosition': [1, 3, 2, 4, 5],
            'Status': ['Finished', 'Finished', 'Finished', 'Finished', 'Finished'],
            'TrackName': ['monza', 'monza', 'monza', 'monza', 'monza'],
            'Year': [2023, 2023, 2023, 2023, 2023],
            'GrandPrix': ['Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix'],
            'Date': [datetime(2023, 9, 3), datetime(2023, 9, 3), datetime(2023, 9, 3), datetime(2023, 9, 3), datetime(2023, 9, 3)],
            'Q1': [pd.Timedelta(seconds=80.123), pd.Timedelta(seconds=80.456), pd.Timedelta(seconds=80.789), 
                   pd.Timedelta(seconds=81.012), pd.Timedelta(seconds=81.345)],
            'Q2': [pd.Timedelta(seconds=79.123), pd.Timedelta(seconds=79.456), pd.Timedelta(seconds=79.789), 
                   pd.Timedelta(seconds=80.012), pd.Timedelta(seconds=80.345)],
            'Q3': [pd.Timedelta(seconds=78.123), pd.Timedelta(seconds=78.456), pd.Timedelta(seconds=78.789), 
                   pd.Timedelta(seconds=79.012), pd.Timedelta(seconds=79.345)],
            'weather_temp_celsius': [28.5, 28.5, 28.5, 28.5, 28.5],
            'weather_wind_speed_ms': [3.2, 3.2, 3.2, 3.2, 3.2],
            'weather_rain_1h_mm': [0, 0, 0, 0, 0],
            'weather_racing_condition': ['dry', 'dry', 'dry', 'dry', 'dry']
        })
        
        # Create sample historical data
        self.historical_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER', 'VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull', 'Red Bull', 'Mercedes', 'Ferrari', 'Ferrari', 'Red Bull'],
            'Position': [1, 3, 2, 4, 5, 2, 1, 3, 5, 4],
            'Points': [25, 15, 18, 12, 10, 18, 25, 15, 10, 12],
            'GridPosition': [1, 3, 2, 4, 5, 1, 2, 3, 5, 4],
            'Status': ['Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished', 'Finished'],
            'TrackName': ['monza', 'monza', 'monza', 'monza', 'monza', 'spa', 'spa', 'spa', 'spa', 'spa'],
            'Year': [2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
            'GrandPrix': ['Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix',
                         'Belgian Grand Prix', 'Belgian Grand Prix', 'Belgian Grand Prix', 'Belgian Grand Prix', 'Belgian Grand Prix'],
            'Date': [datetime(2022, 9, 11), datetime(2022, 9, 11), datetime(2022, 9, 11), datetime(2022, 9, 11), datetime(2022, 9, 11),
                    datetime(2022, 8, 28), datetime(2022, 8, 28), datetime(2022, 8, 28), datetime(2022, 8, 28), datetime(2022, 8, 28)],
            'weather_racing_condition': ['dry', 'dry', 'dry', 'dry', 'dry', 'wet', 'wet', 'wet', 'wet', 'wet'],
            'weather_is_dry': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            'weather_is_any_wet': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        
        # Create sample tire data
        self.tyre_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER', 'VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Stint': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'Compound': ['SOFT', 'SOFT', 'SOFT', 'SOFT', 'SOFT', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'HARD'],
            'LapNumber_min': [1, 1, 1, 1, 1, 25, 25, 25, 25, 25],
            'LapNumber_max': [24, 24, 24, 24, 24, 53, 53, 53, 53, 53],
            'LapNumber_count': [24, 24, 24, 24, 24, 29, 29, 29, 29, 29],
            'TyreLife_max': [24, 24, 24, 24, 24, 29, 29, 29, 29, 29]
        })
    
    def test_create_grid_position_features(self):
        """Test creation of grid position features."""
        # Apply feature engineering
        result = self.engineer.create_grid_position_features(self.race_data)
        
        # Check results
        self.assertTrue('grid_front_row' in result.columns)
        self.assertTrue('grid_top3' in result.columns)
        self.assertTrue('grid_top5' in result.columns)
        self.assertTrue('grid_top10' in result.columns)
        self.assertTrue('grid_group' in result.columns)
        
        # Check values
        self.assertEqual(result.loc[0, 'grid_front_row'], 1)  # VER is on the front row
        self.assertEqual(result.loc[1, 'grid_front_row'], 0)  # HAM is not on the front row
        self.assertEqual(result.loc[0, 'grid_group'], 'pole')  # VER is on pole
    
    def test_create_qualifying_features(self):
        """Test creation of qualifying features."""
        # Apply feature engineering
        result = self.engineer.create_qualifying_features(self.race_data)
        
        # Check results
        self.assertTrue('Q1_seconds' in result.columns)
        self.assertTrue('Q2_seconds' in result.columns)
        self.assertTrue('Q3_seconds' in result.columns)
        self.assertTrue('q_sessions_completed' in result.columns)
        
        # Check values
        self.assertEqual(result.loc[0, 'q_sessions_completed'], 3)  # VER completed all sessions
        self.assertAlmostEqual(result.loc[0, 'Q1_seconds'], 80.123, places=3)
    
    def test_create_team_performance_features(self):
        """Test creation of team performance features."""
        # Apply feature engineering
        result = self.engineer.create_team_performance_features(self.race_data, self.historical_data)
        
        # Check results
        self.assertTrue('team_avg_grid' in result.columns)
        self.assertTrue('grid_vs_teammate' in result.columns)
        
        # Check values
        # VER and PER are teammates, average grid is (1+5)/2 = 3
        # HAM is alone, average grid is 3
        # LEC and SAI are teammates, average grid is (2+4)/2 = 3
        self.assertAlmostEqual(result.loc[0, 'team_avg_grid'], 3.0, places=1)
        self.assertAlmostEqual(result.loc[0, 'grid_vs_teammate'], -2.0, places=1)  # VER is 2 places ahead of average
    
    def test_create_driver_performance_features(self):
        """Test creation of driver performance features."""
        # Apply feature engineering
        result = self.engineer.create_driver_performance_features(self.race_data, self.historical_data)
        
        # Check results
        self.assertTrue('driver_avg_points' in result.columns)
        self.assertTrue('driver_avg_positions_gained' in result.columns)
        self.assertTrue('driver_finish_rate' in result.columns)
        
        # Check values
        # VER's average points in historical data: (25+18)/2 = 21.5
        self.assertAlmostEqual(result.loc[0, 'driver_avg_points'], 21.5, places=1)
        
        # VER's finish rate should be 1.0 (finished all races)
        self.assertAlmostEqual(result.loc[0, 'driver_finish_rate'], 1.0, places=1)
    
    def test_create_circuit_features(self):
        """Test creation of circuit features."""
        # Apply feature engineering
        result = self.engineer.create_circuit_features(self.race_data, self.historical_data)
        
        # Check results
        self.assertTrue('circuit_street' in result.columns)
        self.assertTrue('circuit_high_speed' in result.columns)
        self.assertTrue('circuit_technical' in result.columns)
        
        # Check values - Monza is a high-speed circuit
        self.assertEqual(result.loc[0, 'circuit_high_speed'], 1)
        self.assertEqual(result.loc[0, 'circuit_street'], 0)
    
    def test_create_weather_impact_features(self):
        """Test creation of weather impact features."""
        # Apply feature engineering
        result = self.engineer.create_weather_impact_features(self.race_data, self.historical_data)
        
        # Check results
        self.assertTrue('weather_is_dry' in result.columns)
        self.assertTrue('weather_is_any_wet' in result.columns)
        
        # Check values - race condition is dry
        self.assertEqual(result.loc[0, 'weather_is_dry'], 1)
        self.assertEqual(result.loc[0, 'weather_is_any_wet'], 0)
    
    def test_create_race_strategy_features(self):
        """Test creation of race strategy features."""
        # Apply feature engineering
        result = self.engineer.create_race_strategy_features(self.race_data, self.tyre_data)
        
        # Check results
        self.assertTrue('circuit_high_degradation' in result.columns)
        self.assertTrue('circuit_low_degradation' in result.columns)
        
        # Check values (if we have tire data)
        if 'PitStops' in result.columns:
            # Each driver has 2 stints in our test data
            self.assertEqual(result.loc[0, 'PitStops'], 2)
    
    def test_encode_categorical_features(self):
        """Test encoding of categorical features."""
        # Create test data with categorical features
        test_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC'],
            'Team': ['Red Bull', 'Mercedes', 'Ferrari'],
            'weather_racing_condition': ['dry', 'wet', 'dry']
        })
        
        # Apply encoding
        result = self.engineer.encode_categorical_features(test_data, ['weather_racing_condition'])
        
        # Check results
        self.assertTrue('weather_racing_condition_dry' in result.columns)
        self.assertTrue('weather_racing_condition_wet' in result.columns)
        self.assertFalse('weather_racing_condition' in result.columns)  # Original column should be dropped
        
        # Check values
        self.assertEqual(result.loc[0, 'weather_racing_condition_dry'], 1)
        self.assertEqual(result.loc[0, 'weather_racing_condition_wet'], 0)
    
    def test_scale_numerical_features(self):
        """Test scaling of numerical features."""
        # Create test data with numerical features
        test_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC'],
            'Value1': [10, 20, 30],
            'Value2': [100, 200, 300]
        })
        
        # Apply scaling
        result = self.engineer.scale_numerical_features(test_data, ['Value1', 'Value2'])
        
        # Check results - values should be scaled
        self.assertNotEqual(result.loc[0, 'Value1'], 10)
        self.assertNotEqual(result.loc[0, 'Value2'], 100)
        
        # Mean should be close to 0 and std close to 1 for StandardScaler
        self.assertAlmostEqual(result['Value1'].mean(), 0, places=1)
        self.assertAlmostEqual(result['Value1'].std(), 1, places=1)
    
    def test_create_interaction_features(self):
        """Test creation of interaction features."""
        # Create test data with features for interaction
        test_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC'],
            'GridPosition': [1, 3, 2],
            'overtaking_difficulty': [0.7, 0.7, 0.7],
            'weather_is_any_wet': [0, 1, 0],
            'driver_wet_advantage': [0.5, 1.2, -0.3],
            'circuit_high_speed': [1, 1, 1],
            'team_high_speed_circuit_avg_pos': [2.5, 3.1, 2.8]
        })
        
        # Apply feature engineering
        result = self.engineer.create_interaction_features(test_data)
        
        # Check results
        self.assertTrue('grid_overtaking_interaction' in result.columns)
        self.assertTrue('wet_driver_advantage_interaction' in result.columns)
        self.assertTrue('team_highspeed_interaction' in result.columns)
        
        # Check values
        # grid_overtaking_interaction = GridPosition * overtaking_difficulty
        self.assertAlmostEqual(result.loc[0, 'grid_overtaking_interaction'], 1 * 0.7, places=1)
        
        # wet_driver_advantage_interaction = weather_is_any_wet * driver_wet_advantage
        self.assertAlmostEqual(result.loc[1, 'wet_driver_advantage_interaction'], 1 * 1.2, places=1)
        self.assertAlmostEqual(result.loc[0, 'wet_driver_advantage_interaction'], 0 * 0.5, places=1)
        
        # team_highspeed_interaction = circuit_high_speed * team_high_speed_circuit_avg_pos
        self.assertAlmostEqual(result.loc[0, 'team_highspeed_interaction'], 1 * 2.5, places=1)
    
    def test_create_all_features(self):
        """Test creation of all features."""
        # Apply all feature engineering
        result = self.engineer.create_all_features(
            self.race_data, 
            self.historical_data, 
            self.tyre_data,
            encode_categorical=True
        )
        
        # Check results - should have many more columns than original data
        self.assertGreater(result.shape[1], self.race_data.shape[1])
        
        # Check that key feature groups are present
        grid_features = [col for col in result.columns if col.startswith('grid_')]
        self.assertTrue(len(grid_features) > 0)
        
        qualifying_features = [col for col in result.columns if col.startswith('q_') or col.startswith('Q')]
        self.assertTrue(len(qualifying_features) > 0)
        
        driver_features = [col for col in result.columns if col.startswith('driver_')]
        self.assertTrue(len(driver_features) > 0)
        
        team_features = [col for col in result.columns if col.startswith('team_')]
        self.assertTrue(len(team_features) > 0)
        
        circuit_features = [col for col in result.columns if col.startswith('circuit_')]
        self.assertTrue(len(circuit_features) > 0)
        
        weather_features = [col for col in result.columns if col.startswith('weather_')]
        self.assertTrue(len(weather_features) > 0)
        
        interaction_features = [
            'grid_overtaking_interaction',
            'wet_driver_advantage_interaction',
            'team_highspeed_interaction'
        ]
        # At least some interaction features should be present
        self.assertTrue(any(feat in result.columns for feat in interaction_features))


if __name__ == '__main__':
    unittest.main()