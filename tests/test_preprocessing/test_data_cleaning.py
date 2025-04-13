"""
Tests for the data cleaning module of the F1 prediction project.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from preprocessing.data_cleaning import F1DataCleaner

class TestF1DataCleaner(unittest.TestCase):
    """Test cases for the F1DataCleaner class."""
    
    def setUp(self):
        """Set up test data and cleaner instance."""
        self.cleaner = F1DataCleaner()
        
        # Create sample race data
        self.race_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['Red Bull', 'MERCEDES', 'Ferrari', 'FER', 'RBR'],
            'Position': [1, 2, 3, 4, 5],
            'Points': [25, 18, 15, 12, 10],
            'GridPosition': [1, 3, 2, 4, 5],
            'Status': ['Finished', 'Finished', 'Finished', 'Finished', 'Finished'],
            'TrackName': ['MONZA', 'monza', 'MONZA', 'MONZA', 'MONZA'],
            'Year': [2023, 2023, 2023, 2023, 2023],
            'GrandPrix': ['Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix'],
            'Date': [datetime(2023, 9, 3), datetime(2023, 9, 3), datetime(2023, 9, 3), datetime(2023, 9, 3), datetime(2023, 9, 3)]
        })
        
        # Create sample qualifying data
        self.qualifying_data = pd.DataFrame({
            'Driver': ['VER', 'LEC', 'HAM', 'SAI', 'PER'],
            'Team': ['Red Bull', 'FERRARI', 'Mercedes', 'Ferrari', 'Red Bull'],
            'Position': [1, 2, 3, 4, 5],
            'Q1': [pd.Timedelta(seconds=80.123), pd.Timedelta(seconds=80.456), pd.Timedelta(seconds=80.789), 
                   pd.Timedelta(seconds=81.012), pd.Timedelta(seconds=81.345)],
            'Q2': [pd.Timedelta(seconds=79.123), pd.Timedelta(seconds=79.456), pd.Timedelta(seconds=79.789), 
                   pd.Timedelta(seconds=80.012), pd.Timedelta(seconds=80.345)],
            'Q3': [pd.Timedelta(seconds=78.123), pd.Timedelta(seconds=78.456), pd.Timedelta(seconds=78.789), 
                   pd.Timedelta(seconds=79.012), pd.Timedelta(seconds=79.345)],
            'TrackName': ['MONZA', 'MONZA', 'MONZA', 'MONZA', 'MONZA'],
            'Year': [2023, 2023, 2023, 2023, 2023],
            'GrandPrix': ['Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix', 'Italian Grand Prix'],
        })
        
        # Create sample weather data
        self.weather_data = pd.DataFrame({
            'timestamp': [datetime(2023, 9, 3, 14, 0), datetime(2023, 9, 3, 14, 30), datetime(2023, 9, 3, 15, 0)],
            'temp_celsius': [28.5, 29.2, 29.8],
            'feels_like_celsius': [30.1, 30.9, 31.5],
            'humidity_percent': [45, 43, 40],
            'wind_speed_ms': [3.2, 3.5, 3.8],
            'wind_gust_ms': [5.1, 5.6, 6.2],
            'rain_1h_mm': [0, 0, 0],
            'clouds_percent': [10, 15, 20],
            'weather_category': ['clear', 'clear', 'clouds_few'],
            'racing_condition': ['dry', 'dry', 'dry'],
            'circuit': ['monza', 'monza', 'monza']
        })
        
        # Create sample lap data
        self.lap_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['Red Bull', 'MERCEDES', 'Ferrari', 'FER', 'RBR'],
            'LapNumber': [1, 1, 1, 1, 1],
            'LapTime': [pd.Timedelta(seconds=82.123), pd.Timedelta(seconds=82.456), pd.Timedelta(seconds=82.789), 
                        pd.Timedelta(seconds=83.012), pd.Timedelta(seconds=83.345)],
            'Stint': [1, 1, 1, 1, 1],
            'Compound': ['SOFT', 'SOFT', 'SOFT', 'SOFT', 'SOFT'],
            'TrackName': ['MONZA', 'monza', 'MONZA', 'MONZA', 'MONZA'],
        })
        
        # Create sample telemetry data
        self.telemetry_data = {
            'car_data': pd.DataFrame({
                'Speed': [320, 325, 330, 335, 340],
                'RPM': [12000, 12100, 12200, 12300, 12400],
                'nGear': [8, 8, 8, 8, 8],
                'Throttle': [100, 100, 100, 100, 100],
                'Brake': [0, 0, 0, 0, 0],
                'DRS': [1, 1, 1, 1, 1],
                'Time': [pd.Timedelta(seconds=0.1), pd.Timedelta(seconds=0.2), pd.Timedelta(seconds=0.3), 
                         pd.Timedelta(seconds=0.4), pd.Timedelta(seconds=0.5)],
                'LapNumber': [1, 1, 1, 1, 1]
            }),
            'pos_data': pd.DataFrame({
                'X': [100, 110, 120, 130, 140],
                'Y': [200, 210, 220, 230, 240],
                'Z': [10, 10, 10, 10, 10],
                'Time': [pd.Timedelta(seconds=0.1), pd.Timedelta(seconds=0.2), pd.Timedelta(seconds=0.3), 
                         pd.Timedelta(seconds=0.4), pd.Timedelta(seconds=0.5)],
                'LapNumber': [1, 1, 1, 1, 1]
            })
        }
    
    def test_standardize_driver_names(self):
        """Test standardization of driver names."""
        # Create test data with different driver name formats
        test_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'MAX', 'LEW', 'C LECLERC', 'L HAMILTON']
        })
        
        # Apply standardization
        cleaned_data = self.cleaner.standardize_driver_names(test_data)
        
        # Check results
        expected_drivers = ['VER', 'HAM', 'VER', 'HAM', 'LEC', 'HAM']
        self.assertListEqual(cleaned_data['Driver'].tolist(), expected_drivers)
    
    def test_standardize_team_names(self):
        """Test standardization of team names."""
        # Create test data with different team name formats
        test_data = pd.DataFrame({
            'Team': ['RED BULL', 'REDBULL', 'Ferrari', 'FERRARI', 'MERCEDES', 'Mercedes-AMG']
        })
        
        # Apply standardization
        cleaned_data = self.cleaner.standardize_team_names(test_data)
        
        # Check results
        expected_teams = ['Red Bull', 'Red Bull', 'Ferrari', 'Ferrari', 'Mercedes', 'Mercedes']
        self.assertListEqual(cleaned_data['Team'].tolist(), expected_teams)
    
    def test_standardize_circuit_names(self):
        """Test standardization of circuit names."""
        # Create test data with different circuit name formats
        test_data = pd.DataFrame({
            'Circuit': ['MONZA', 'SPA-FRANCORCHAMPS', 'MONACO', 'SILVERSTONE', 'BAHRAIN']
        })
        
        # Apply standardization
        cleaned_data = self.cleaner.standardize_circuit_names(test_data)
        
        # Check results
        expected_circuits = ['monza', 'spa', 'monaco', 'silverstone', 'bahrain']
        self.assertListEqual(cleaned_data['Circuit'].tolist(), expected_circuits)
    
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        # Create test data with missing values
        test_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', np.nan],
            'Position': [1, 2, np.nan, 4, 5],
            'Points': [25, np.nan, 15, 12, 10]
        })
        
        # Apply missing value handling
        cleaned_data = self.cleaner.handle_missing_values(test_data, strategy='mean')
        
        # Check results
        self.assertEqual(cleaned_data['Position'].isna().sum(), 0)
        self.assertEqual(cleaned_data['Points'].isna().sum(), 0)
        # Position 3 should be filled with mean of [1,2,4,5]
        self.assertAlmostEqual(cleaned_data.loc[2, 'Points'], 15)
        # Points for HAM should be filled with mean of [25,15,12,10]
        self.assertAlmostEqual(cleaned_data.loc[1, 'Points'], (25+15+12+10)/4)
    
    def test_handle_outliers(self):
        """Test handling of outliers."""
        # Create test data with outliers
        test_data = pd.DataFrame({
            'Value': [10, 12, 11, 13, 50, 9, 14]  # 50 is an outlier
        })
        
        # Apply outlier handling
        cleaned_data = self.cleaner.handle_outliers(test_data, method='clip')
        
        # Check results - 50 should be clipped
        self.assertTrue(cleaned_data['Value'].max() < 50)
    
    def test_merge_race_qualifying_data(self):
        """Test merging of race and qualifying data."""
        # Apply merge
        merged_data = self.cleaner.merge_race_qualifying_data(self.race_data, self.qualifying_data)
        
        # Check results
        self.assertEqual(len(merged_data), len(self.race_data))
        self.assertTrue('Position' in merged_data.columns)
        self.assertTrue('Q_Position' in merged_data.columns)
        self.assertTrue('Q1' in merged_data.columns or 'Q_Q1' in merged_data.columns)
    
    def test_merge_with_weather_data(self):
        """Test merging with weather data."""
        # Apply merge
        merged_data = self.cleaner.merge_with_weather_data(self.race_data, self.weather_data)
        
        # Check results
        self.assertEqual(len(merged_data), len(self.race_data))
        weather_cols = [col for col in merged_data.columns if col.startswith('weather_')]
        self.assertTrue(len(weather_cols) > 0)
    
    def test_clean_lap_data(self):
        """Test cleaning of lap data."""
        # Apply cleaning
        cleaned_data = self.cleaner.clean_lap_data(self.lap_data)
        
        # Check results
        self.assertEqual(len(cleaned_data), len(self.lap_data))
        self.assertTrue('LapTime_seconds' in cleaned_data.columns)
    
    def test_clean_telemetry_data(self):
        """Test cleaning of telemetry data."""
        # Apply cleaning
        cleaned_data = self.cleaner.clean_telemetry_data(self.telemetry_data)
        
        # Check results
        self.assertTrue('car_data' in cleaned_data)
        self.assertTrue('pos_data' in cleaned_data)
        self.assertEqual(len(cleaned_data['car_data']), len(self.telemetry_data['car_data']))
    
    def test_prepare_modeling_dataset(self):
        """Test preparation of modeling dataset."""
        # Apply preparation
        prepared_data = self.cleaner.prepare_modeling_dataset(
            self.race_data, self.qualifying_data, self.weather_data
        )
        
        # Check results
        self.assertEqual(len(prepared_data), len(self.race_data))
        # Should have columns from all datasets
        self.assertTrue('Position' in prepared_data.columns)
        self.assertTrue('Q1' in prepared_data.columns or 'Q_Q1' in prepared_data.columns)
        
        weather_cols = [col for col in prepared_data.columns if col.startswith('weather_')]
        self.assertTrue(len(weather_cols) > 0)
    
    def test_convert_lap_time_to_seconds(self):
        """Test conversion of lap times to seconds."""
        # Test with different formats
        time_delta = pd.Timedelta(seconds=82.123)
        string_format = "1:22.123"
        float_format = 82.123
        
        # Apply conversion
        seconds1 = self.cleaner.convert_lap_time_to_seconds(time_delta)
        seconds2 = self.cleaner.convert_lap_time_to_seconds(string_format)
        seconds3 = self.cleaner.convert_lap_time_to_seconds(float_format)
        
        # Check results
        self.assertAlmostEqual(seconds1, 82.123, places=3)
        self.assertAlmostEqual(seconds2, 82.123, places=3)
        self.assertAlmostEqual(seconds3, 82.123, places=3)
    
    def test_clean_race_weekend_data(self):
        """Test cleaning of race weekend data."""
        # Apply cleaning
        cleaned_data = self.cleaner.clean_race_weekend_data(
            [self.lap_data], self.qualifying_data, self.race_data
        )
        
        # Check results
        self.assertTrue('Practice1' in cleaned_data)
        self.assertTrue('Qualifying' in cleaned_data)
        self.assertTrue('Race' in cleaned_data)
        self.assertEqual(len(cleaned_data['Race']), len(self.race_data))


if __name__ == '__main__':
    unittest.main()