"""
Integration tests for the preprocessing pipeline of the F1 prediction project.
Tests both data cleaning and feature engineering together.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from preprocessing.data_cleaning import F1DataCleaner
from preprocessing.feature_engineering import F1FeatureEngineer

class TestPreprocessingIntegration(unittest.TestCase):
    """Integration test cases for the preprocessing pipeline."""
    
    def setUp(self):
        """Set up test data and instances."""
        self.cleaner = F1DataCleaner()
        self.engineer = F1FeatureEngineer(scale_features=True, scaling_method='standard')
        
        # Create sample race data
        self.race_data = pd.DataFrame({
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER'],
            'Team': ['RED BULL', 'MERCEDES', 'Ferrari', 'FER', 'RBR'],
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
    
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline, from raw data to feature engineering."""
        # Step 1: Clean and merge the raw data
        merged_data = self.cleaner.merge_race_qualifying_data(self.race_data, self.qualifying_data)
        merged_data_with_weather = self.cleaner.merge_with_weather_data(merged_data, self.weather_data)
        
        # Clean historical data
        historical_clean = self.cleaner.standardize_driver_names(self.historical_data)
        historical_clean = self.cleaner.standardize_team_names(historical_clean)
        historical_clean = self.cleaner.standardize_circuit_names(historical_clean, 'TrackName')
        
        # Step 2: Apply feature engineering
        engineered_data = self.engineer.create_all_features(
            merged_data_with_weather,
            historical_clean,
            encode_categorical=True
        )
        
        # Verify that the pipeline ran successfully
        self.assertIsNotNone(engineered_data)
        self.assertGreater(len(engineered_data), 0)
        
        # Verify that the final dataset has all the expected feature categories
        feature_categories = [
            'grid_', 'Q', 'q_', 'driver_', 'team_', 'circuit_', 'weather_'
        ]
        
        for category in feature_categories:
            category_features = [col for col in engineered_data.columns if col.startswith(category)]
            self.assertTrue(len(category_features) > 0, f"No features found for category: {category}")
        
        # Verify that all drivers are present in the final dataset
        for driver in self.race_data['Driver'].unique():
            self.assertTrue(driver in engineered_data['Driver'].values)
    
    def test_prediction_data_preparation(self):
        """Test preparing data for making predictions (without known results)."""
        # Create a dataset that simulates pre-race data (qualifying results but no race results)
        pre_race_data = self.qualifying_data.copy()
        pre_race_data['GridPosition'] = pre_race_data['Position']
        pre_race_data = pre_race_data.drop('Position', axis=1)
        
        # Clean and merge with weather
        cleaned_data = self.cleaner.standardize_driver_names(pre_race_data)
        cleaned_data = self.cleaner.standardize_team_names(cleaned_data)
        cleaned_data = self.cleaner.standardize_circuit_names(cleaned_data, 'TrackName')
        
        merged_data = self.cleaner.merge_with_weather_data(cleaned_data, self.weather_data)
        
        # Clean historical data
        historical_clean = self.cleaner.standardize_driver_names(self.historical_data)
        historical_clean = self.cleaner.standardize_team_names(historical_clean)
        historical_clean = self.cleaner.standardize_circuit_names(historical_clean, 'TrackName')
        
        # Apply feature engineering
        engineered_data = self.engineer.create_all_features(
            merged_data,
            historical_clean,
            encode_categorical=True
        )
        
        # Verify that the preparation ran successfully
        self.assertIsNotNone(engineered_data)
        self.assertGreater(len(engineered_data), 0)
        
        # Verify we have features to make predictions
        prediction_features = [
            'grid_top3', 'grid_front_row', 
            'driver_avg_points', 'driver_finish_rate',
            'team_avg_grid', 'circuit_high_speed',
            'weather_is_dry'
        ]
        
        for feature in prediction_features:
            self.assertTrue(
                feature in engineered_data.columns or 
                any(col.startswith(feature) for col in engineered_data.columns),
                f"Missing prediction feature: {feature}"
            )
    
    def test_handling_missing_data(self):
        """Test the pipeline's ability to handle missing data."""
        # Create data with missing values
        race_with_missing = self.race_data.copy()
        race_with_missing.loc[0, 'GridPosition'] = np.nan
        race_with_missing.loc[1, 'Points'] = np.nan
        
        qualifying_with_missing = self.qualifying_data.copy()
        qualifying_with_missing.loc[0, 'Q1'] = np.nan
        qualifying_with_missing.loc[1, 'Q3'] = np.nan
        
        # Run the pipeline
        merged_data = self.cleaner.merge_race_qualifying_data(race_with_missing, qualifying_with_missing)
        merged_data_with_weather = self.cleaner.merge_with_weather_data(merged_data, self.weather_data)
        
        engineered_data = self.engineer.create_all_features(
            merged_data_with_weather,
            self.historical_data,
            encode_categorical=True
        )
        
        # Verify that the pipeline handled missing values
        self.assertIsNotNone(engineered_data)
        self.assertGreater(len(engineered_data), 0)
        
        # The final dataset should have no missing values in key features
        numerical_features = engineered_data.select_dtypes(include=['number']).columns
        self.assertEqual(engineered_data[numerical_features].isna().sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()