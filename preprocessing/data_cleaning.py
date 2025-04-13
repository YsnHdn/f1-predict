"""
Data cleaning module for F1 prediction project.
This module handles cleaning and merging data from FastF1 and weather sources.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class F1DataCleaner:
    """
    Cleans and prepares F1 data for modeling.
    Handles merging of race data, qualifying data, and weather data.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        logger.info("Initializing F1 data cleaner")
    
    def standardize_driver_names(self, df: pd.DataFrame, name_col: str = 'Driver') -> pd.DataFrame:
        """
        Standardize driver names/codes to ensure consistency across datasets.
        
        Args:
            df: DataFrame containing driver information
            name_col: Column name that contains driver identifiers
            
        Returns:
            DataFrame with standardized driver names
        """
        if name_col not in df.columns:
            logger.warning(f"Column {name_col} not found in DataFrame")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Mapping for potential inconsistencies in driver codes
        driver_code_mapping = {
            # Common variations in driver codes
            'VER': 'VER', 'MAX': 'VER', 'M VERSTAPPEN': 'VER',
            'HAM': 'HAM', 'LEW': 'HAM', 'L HAMILTON': 'HAM',
            'PER': 'PER', 'CHE': 'PER', 'S PEREZ': 'PER',
            'LEC': 'LEC', 'CHA': 'LEC', 'C LECLERC': 'LEC',
            'SAI': 'SAI', 'CAR': 'SAI', 'C SAINZ': 'SAI', 
            'NOR': 'NOR', 'LAN': 'NOR', 'L NORRIS': 'NOR',
            'PIA': 'PIA', 'OSC': 'PIA', 'O PIASTRI': 'PIA',
            'RUS': 'RUS', 'GEO': 'RUS', 'G RUSSELL': 'RUS',
            'ALO': 'ALO', 'FER': 'ALO', 'F ALONSO': 'ALO',
            'STR': 'STR', 'LAN': 'STR', 'L STROLL': 'STR',
            'OCO': 'OCO', 'EST': 'OCO', 'E OCON': 'OCO',
            'GAS': 'GAS', 'PIE': 'GAS', 'P GASLY': 'GAS',
            'ALB': 'ALB', 'ALE': 'ALB', 'A ALBON': 'ALB',
            'SAR': 'SAR', 'LOG': 'SAR', 'L SARGEANT': 'SAR',
            'TSU': 'TSU', 'YUK': 'TSU', 'Y TSUNODA': 'TSU',
            'BOT': 'BOT', 'VAL': 'BOT', 'V BOTTAS': 'BOT',
            'ZHO': 'ZHO', 'GUA': 'ZHO', 'G ZHOU': 'ZHO',
            'MAG': 'MAG', 'KEV': 'MAG', 'K MAGNUSSEN': 'MAG',
            'HUL': 'HUL', 'NIC': 'HUL', 'N HULKENBERG': 'HUL'
            # Add more mappings as needed
        }
        
        # Apply standardization
        result_df[name_col] = result_df[name_col].map(
            lambda x: driver_code_mapping.get(str(x).upper(), x) if pd.notnull(x) else x
        )
        
        logger.info(f"Standardized driver names in column {name_col}")
        return result_df
    
    def standardize_team_names(self, df: pd.DataFrame, team_col: str = 'Team') -> pd.DataFrame:
        """
        Standardize team names to ensure consistency across datasets.
        
        Args:
            df: DataFrame containing team information
            team_col: Column name that contains team identifiers
            
        Returns:
            DataFrame with standardized team names
        """
        if team_col not in df.columns:
            logger.warning(f"Column {team_col} not found in DataFrame")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Mapping for potential inconsistencies in team names
        team_name_mapping = {
            # Current teams with variations
            'RED BULL': 'Red Bull', 'REDBULL': 'Red Bull', 'RBR': 'Red Bull', 'RED BULL RACING': 'Red Bull',
            'FERRARI': 'Ferrari', 'FER': 'Ferrari', 'SCUDERIA FERRARI': 'Ferrari',
            'MERCEDES': 'Mercedes', 'MERC': 'Mercedes', 'MER': 'Mercedes', 'MERCEDES-AMG': 'Mercedes',
            'MCLAREN': 'McLaren', 'MCL': 'McLaren', 'MAC': 'McLaren',
            'ASTON MARTIN': 'Aston Martin', 'ASTONMARTIN': 'Aston Martin', 'AST': 'Aston Martin', 'AM': 'Aston Martin',
            'ALPINE': 'Alpine', 'ALP': 'Alpine', 
            'WILLIAMS': 'Williams', 'WIL': 'Williams',
            'ALPHA TAURI': 'RB', 'ALPHATAURI': 'RB', 'AT': 'RB',
            'RACING BULLS': 'RB', 'RB': 'RB', 
            'SAUBER': 'Sauber', 'SAU': 'Sauber', 'ALFA ROMEO': 'Sauber',
            'HAAS': 'Haas', 'HAS': 'Haas', 'HAAS F1': 'Haas'
            # Add more mappings as needed
        }
        
        # Apply standardization
        result_df[team_col] = result_df[team_col].map(
            lambda x: team_name_mapping.get(str(x).upper(), x) if pd.notnull(x) else x
        )
        
        logger.info(f"Standardized team names in column {team_col}")
        return result_df
    
    def standardize_circuit_names(self, df: pd.DataFrame, circuit_col: str = 'Circuit') -> pd.DataFrame:
        """
        Standardize circuit names to ensure consistency between FastF1 and weather data.
        
        Args:
            df: DataFrame containing circuit information
            circuit_col: Column name that contains circuit identifiers
            
        Returns:
            DataFrame with standardized circuit names
        """
        if circuit_col not in df.columns:
            logger.warning(f"Column {circuit_col} not found in DataFrame")
            return df
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Mapping between FastF1 circuit names and weather API circuit keys
        circuit_name_mapping = {
            # FastF1 name to standardized key
            'BAHRAIN': 'bahrain', 'SAKHIR': 'bahrain',
            'JEDDAH': 'jeddah', 'SAUDI ARABIA': 'jeddah',
            'ALBERT PARK': 'albert_park', 'MELBOURNE': 'albert_park', 'AUSTRALIA': 'albert_park',
            'IMOLA': 'imola', 'EMILIA ROMAGNA': 'imola',
            'MIAMI': 'miami',
            'MONACO': 'monaco', 'MONTE CARLO': 'monaco',
            'BARCELONA': 'barcelona', 'CATALUNYA': 'barcelona', 'SPAIN': 'barcelona',
            'MONTREAL': 'montreal', 'CANADA': 'montreal',
            'SILVERSTONE': 'silverstone', 'GREAT BRITAIN': 'silverstone',
            'SPIELBERG': 'spielberg', 'RED BULL RING': 'spielberg', 'AUSTRIA': 'spielberg',
            'PAUL RICARD': 'paul_ricard', 'FRANCE': 'paul_ricard',
            'HUNGARORING': 'hungaroring', 'HUNGARY': 'hungaroring',
            'SPA': 'spa', 'SPA-FRANCORCHAMPS': 'spa', 'BELGIUM': 'spa',
            'ZANDVOORT': 'zandvoort', 'NETHERLANDS': 'zandvoort',
            'MONZA': 'monza', 'ITALY': 'monza',
            'BAKU': 'baku', 'AZERBAIJAN': 'baku',
            'MARINA BAY': 'marina_bay', 'SINGAPORE': 'marina_bay',
            'SUZUKA': 'suzuka', 'JAPAN': 'suzuka',
            'AMERICAS': 'americas', 'COTA': 'americas', 'AUSTIN': 'americas',
            'RODRIGUEZ': 'rodriguez', 'MEXICO CITY': 'rodriguez', 'MEXICO': 'rodriguez',
            'INTERLAGOS': 'interlagos', 'SAO PAULO': 'interlagos', 'BRAZIL': 'interlagos',
            'LAS VEGAS': 'las_vegas',
            'LOSAIL': 'losail', 'QATAR': 'losail',
            'YAS MARINA': 'yas_marina', 'ABU DHABI': 'yas_marina',
            'SHANGHAI': 'shanghai', 'CHINA': 'shanghai',
            'PORTIMAO': 'portimao', 'PORTUGAL': 'portimao',
            'ISTANBUL': 'istanbul', 'TURKEY': 'istanbul',
            'SOCHI': 'sochi', 'RUSSIA': 'sochi',
            'NURBURGRING': 'nurburgring', 'GERMANY': 'nurburgring',
            'MUGELLO': 'mugello'
            # Add more mappings as needed
        }
        
        # Apply standardization
        result_df[circuit_col] = result_df[circuit_col].map(
            lambda x: circuit_name_mapping.get(str(x).upper(), x.lower()) if pd.notnull(x) else x
        )
        
        logger.info(f"Standardized circuit names in column {circuit_col}")
        return result_df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values in the DataFrame.
        
        Args:
            df: DataFrame with missing values
            strategy: Strategy to handle missing values ('mean', 'median', 'mode', 'drop', 'zero')
            
        Returns:
            DataFrame with handled missing values
        """
        result_df = df.copy()
        
        # Check for missing values
        missing_count = result_df.isna().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found in DataFrame")
            return result_df
        
        logger.info(f"Handling {missing_count} missing values with strategy: {strategy}")
        
        # Handle missing values based on the strategy
        if strategy == 'drop':
            # Drop rows with any missing values
            result_df = result_df.dropna()
            
        elif strategy == 'zero':
            # Fill numeric columns with 0 and categorical with 'Unknown'
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            categorical_cols = result_df.select_dtypes(exclude=['number']).columns
            
            result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
            result_df[categorical_cols] = result_df[categorical_cols].fillna('Unknown')
            
        elif strategy == 'mean':
            # Fill numeric columns with mean and categorical with mode
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            categorical_cols = result_df.select_dtypes(exclude=['number']).columns
            
            for col in numeric_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mean())
                
            for col in categorical_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else 'Unknown')
                
        elif strategy == 'median':
            # Fill numeric columns with median and categorical with mode
            numeric_cols = result_df.select_dtypes(include=['number']).columns
            categorical_cols = result_df.select_dtypes(exclude=['number']).columns
            
            for col in numeric_cols:
                result_df[col] = result_df[col].fillna(result_df[col].median())
                
            for col in categorical_cols:
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else 'Unknown')
                
        elif strategy == 'mode':
            # Fill all columns with mode
            for col in result_df.columns:
                mode_value = result_df[col].mode()[0] if not result_df[col].mode().empty else None
                if mode_value is not None:
                    result_df[col] = result_df[col].fillna(mode_value)
                else:
                    # Fallback to 0 for numeric and 'Unknown' for others
                    if pd.api.types.is_numeric_dtype(result_df[col]):
                        result_df[col] = result_df[col].fillna(0)
                    else:
                        result_df[col] = result_df[col].fillna('Unknown')
        
        # Check remaining missing values
        remaining_missing = result_df.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"There are still {remaining_missing} missing values after handling")
            
        return result_df
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'clip') -> pd.DataFrame:
        """
        Detect and handle outliers in the dataset.
        
        Args:
            df: DataFrame to process
            columns: List of columns to check for outliers (None means all numeric columns)
            method: Method to handle outliers ('clip', 'remove', 'winsorize')
            
        Returns:
            DataFrame with handled outliers
        """
        result_df = df.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = result_df.select_dtypes(include=['number']).columns.tolist()
        else:
            # Filter to only include numeric columns from the provided list
            columns = [col for col in columns if col in result_df.columns and 
                      pd.api.types.is_numeric_dtype(result_df[col])]
        
        if not columns:
            logger.warning("No valid numeric columns found for outlier detection")
            return result_df
            
        logger.info(f"Handling outliers in {len(columns)} columns using method: {method}")
        
        for col in columns:
            # Calculate Q1, Q3, and IQR
            Q1 = result_df[col].quantile(0.25)
            Q3 = result_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_count = ((result_df[col] < lower_bound) | (result_df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                logger.info(f"Found {outliers_count} outliers in column '{col}'")
                
                if method == 'clip':
                    # Clip values to the bounds
                    result_df[col] = result_df[col].clip(lower_bound, upper_bound)
                    
                elif method == 'remove':
                    # Remove rows with outliers
                    result_df = result_df[~((result_df[col] < lower_bound) | (result_df[col] > upper_bound))]
                    
                elif method == 'winsorize':
                    # Replace outliers with the bounds
                    result_df.loc[result_df[col] < lower_bound, col] = lower_bound
                    result_df.loc[result_df[col] > upper_bound, col] = upper_bound
            else:
                logger.info(f"No outliers found in column '{col}'")
                
        return result_df
    
    def merge_race_qualifying_data(self, race_df: pd.DataFrame, qualifying_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge race results with qualifying data.
        
        Args:
            race_df: DataFrame containing race results
            qualifying_df: DataFrame containing qualifying results
            
        Returns:
            Merged DataFrame with both race and qualifying data
        """
        if race_df is None or qualifying_df is None or race_df.empty or qualifying_df.empty:
            logger.error("Cannot merge: one or both DataFrames are empty or None")
            return pd.DataFrame()
        
        # Standardize driver names
        race_df = self.standardize_driver_names(race_df)
        qualifying_df = self.standardize_driver_names(qualifying_df)
        
        # Make sure we have the correct columns to merge on
        merge_columns = ['Year', 'GrandPrix', 'Driver']
        
        # Verify merge columns exist in both DataFrames
        missing_race_cols = [col for col in merge_columns if col not in race_df.columns]
        missing_qual_cols = [col for col in merge_columns if col not in qualifying_df.columns]
        
        if missing_race_cols or missing_qual_cols:
            logger.error(f"Missing columns for merge: Race DF missing {missing_race_cols}, Qualifying DF missing {missing_qual_cols}")
            return pd.DataFrame()
        
        # Perform the merge
        logger.info("Merging race and qualifying data")
        
        # To avoid duplicate columns, add a prefix to qualifying columns (except merge columns)
        qual_columns = {col: f'Q_{col}' for col in qualifying_df.columns if col not in merge_columns}
        qualifying_df = qualifying_df.rename(columns=qual_columns)
        
        # Merge the DataFrames
        merged_df = pd.merge(race_df, qualifying_df, on=merge_columns, how='left')
        
        logger.info(f"Merged DataFrame has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
        
        # Handle any missing values that might have been introduced in the merge
        merged_df = self.handle_missing_values(merged_df, strategy='mean')
        
        return merged_df
    
    def merge_with_weather_data(self, race_df: pd.DataFrame, weather_df: pd.DataFrame, 
                               race_time_col: str = 'Date', weather_time_col: str = 'timestamp') -> pd.DataFrame:
        """
        Merge race data with weather data, matching on circuit and time.
        
        Args:
            race_df: DataFrame containing race results
            weather_df: DataFrame containing weather data
            race_time_col: Column name in race_df for race datetime
            weather_time_col: Column name in weather_df for weather timestamp
            
        Returns:
            Merged DataFrame with race and weather data
        """
        if race_df is None or weather_df is None or race_df.empty or weather_df.empty:
            logger.error("Cannot merge: one or both DataFrames are empty or None")
            return pd.DataFrame()
        
        # Standardize circuit names
        if 'TrackName' in race_df.columns:
            race_df = self.standardize_circuit_names(race_df, circuit_col='TrackName')
        elif 'Circuit' in race_df.columns:
            race_df = self.standardize_circuit_names(race_df, circuit_col='Circuit')
        else:
            logger.error("No circuit column found in race DataFrame")
            return pd.DataFrame()
            
        if 'circuit' in weather_df.columns:
            weather_df = self.standardize_circuit_names(weather_df, circuit_col='circuit')
        else:
            logger.error("No circuit column found in weather DataFrame")
            return pd.DataFrame()
        
        # Ensure datetime format for matching
        try:
            # Convert race time to datetime if it's not already
            if race_time_col in race_df.columns and not pd.api.types.is_datetime64_dtype(race_df[race_time_col]):
                race_df[race_time_col] = pd.to_datetime(race_df[race_time_col])
                
            # Convert weather time to datetime if it's not already
            if weather_time_col in weather_df.columns and not pd.api.types.is_datetime64_dtype(weather_df[weather_time_col]):
                weather_df[weather_time_col] = pd.to_datetime(weather_df[weather_time_col])
        except Exception as e:
            logger.error(f"Error converting datetime columns: {str(e)}")
            return pd.DataFrame()
        
        # Create a copy of the race DataFrame to modify
        result_df = race_df.copy()
        
        # Get unique circuit-date combinations from race data
        circuit_dates = []
        
        if 'TrackName' in result_df.columns and race_time_col in result_df.columns:
            circuit_dates = result_df[['TrackName', race_time_col]].drop_duplicates().values
        elif 'Circuit' in result_df.columns and race_time_col in result_df.columns:
            circuit_dates = result_df[['Circuit', race_time_col]].drop_duplicates().values
        
        if len(circuit_dates) == 0:
            logger.error("Could not extract circuit-date combinations from race data")
            return pd.DataFrame()
        
        logger.info(f"Found {len(circuit_dates)} unique circuit-date combinations")
        
        # Create weather feature columns in the result DataFrame
        weather_features = [
            'temp_celsius', 'feels_like_celsius', 'humidity_percent', 
            'wind_speed_ms', 'wind_gust_ms', 'rain_1h_mm', 
            'clouds_percent', 'visibility_meters', 'racing_condition'
        ]
        
        for feature in weather_features:
            if feature in weather_df.columns:
                result_df[f'weather_{feature}'] = np.nan
        
        # Match each race with the closest weather data point
        matched_count = 0
        
        for circuit, race_date in circuit_dates:
            # Filter weather data for this circuit
            circuit_weather = None
            
            if 'circuit' in weather_df.columns and weather_time_col in weather_df.columns:
                circuit_weather = weather_df[weather_df['circuit'] == circuit.lower()]
            
            if circuit_weather is None or circuit_weather.empty:
                logger.warning(f"No weather data found for circuit {circuit}")
                continue
                
            # Find the closest weather data point to the race time
            # Allow for a window of ±3 hours
            time_window = 3  # hours
            race_time_min = race_date - pd.Timedelta(hours=time_window)
            race_time_max = race_date + pd.Timedelta(hours=time_window)
            
            closest_weather = circuit_weather[
                (circuit_weather[weather_time_col] >= race_time_min) & 
                (circuit_weather[weather_time_col] <= race_time_max)
            ]
            
            if closest_weather.empty:
                logger.warning(f"No weather data found within {time_window} hours of race at {circuit} on {race_date}")
                continue
                
            # If multiple matches, take the closest to the race time
            if len(closest_weather) > 1:
                closest_weather['time_diff'] = abs(closest_weather[weather_time_col] - race_date)
                closest_weather = closest_weather.sort_values('time_diff').head(1)
            
            # Update the result DataFrame with weather features
            for feature in weather_features:
                if feature in weather_df.columns:
                    # Find rows in result_df matching this circuit and date
                    if 'TrackName' in result_df.columns:
                        mask = (result_df['TrackName'] == circuit) & (result_df[race_time_col] == race_date)
                    else:
                        mask = (result_df['Circuit'] == circuit) & (result_df[race_time_col] == race_date)
                    
                    # Update the weather feature
                    result_df.loc[mask, f'weather_{feature}'] = closest_weather[feature].values[0]
            
            matched_count += 1
            
        logger.info(f"Matched weather data for {matched_count} out of {len(circuit_dates)} circuit-dates")
        
        # Handle any missing weather values
        result_df = self.handle_missing_values(result_df, strategy='mean')
        
        return result_df
    
    def clean_lap_data(self, lap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare lap data for analysis.
        
        Args:
            lap_df: DataFrame containing lap data
            
        Returns:
            Cleaned lap data DataFrame
        """
        if lap_df is None or lap_df.empty:
            logger.error("Cannot clean empty lap DataFrame")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        result_df = lap_df.copy()
        
        # Standardize driver and team names
        result_df = self.standardize_driver_names(result_df)
        if 'Team' in result_df.columns:
            result_df = self.standardize_team_names(result_df)
        
        # Convert lap time from timedelta to seconds for easier numerical analysis
        if 'LapTime' in result_df.columns:
            # Check if LapTime is already numeric
            if not pd.api.types.is_numeric_dtype(result_df['LapTime']):
                try:
                    # Convert to seconds
                    result_df['LapTime_seconds'] = result_df['LapTime'].dt.total_seconds()
                except Exception as e:
                    logger.error(f"Error converting lap times to seconds: {str(e)}")
                    # Create a numeric column anyway
                    result_df['LapTime_seconds'] = np.nan
            else:
                # Already numeric, just copy
                result_df['LapTime_seconds'] = result_df['LapTime']
        
        # Flag outlier laps
        if 'LapTime_seconds' in result_df.columns:
            # Calculate lap time statistics by circuit
            if 'TrackName' in result_df.columns:
                circuit_stats = result_df.groupby('TrackName')['LapTime_seconds'].agg(['mean', 'std']).reset_index()
            elif 'Circuit' in result_df.columns:
                circuit_stats = result_df.groupby('Circuit')['LapTime_seconds'].agg(['mean', 'std']).reset_index()
            else:
                # No circuit column, calculate global stats
                mean_lap = result_df['LapTime_seconds'].mean()
                std_lap = result_df['LapTime_seconds'].std()
                
                # Flag global outliers (±3 standard deviations)
                result_df['lap_outlier'] = (
                    (result_df['LapTime_seconds'] < mean_lap - 3*std_lap) | 
                    (result_df['LapTime_seconds'] > mean_lap + 3*std_lap)
                )
                return result_df
            
            # Join circuit stats back to the lap data
            if 'TrackName' in result_df.columns:
                result_df = pd.merge(result_df, circuit_stats, on='TrackName', how='left')
            else:
                result_df = pd.merge(result_df, circuit_stats, on='Circuit', how='left')
                
            # Flag outliers (±3 standard deviations from circuit mean)
            result_df['lap_outlier'] = (
                (result_df['LapTime_seconds'] < result_df['mean'] - 3*result_df['std']) | 
                (result_df['LapTime_seconds'] > result_df['mean'] + 3*result_df['std'])
            )
            
            # Clean up temporary columns
            result_df = result_df.drop(['mean', 'std'], axis=1)
        
        # Handle missing values
        result_df = self.handle_missing_values(result_df, strategy='mean')
        
        return result_df
    
    def clean_telemetry_data(self, telemetry_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean and prepare telemetry data for analysis.
        
        Args:
            telemetry_dict: Dictionary containing telemetry DataFrames
            
        Returns:
            Dictionary with cleaned telemetry DataFrames
        """
        if not telemetry_dict:
            logger.error("Cannot clean empty telemetry data")
            return {}
        
        result_dict = {}
        
        # Clean each DataFrame in the dictionary
        for key, df in telemetry_dict.items():
            if df is None or df.empty:
                logger.warning(f"Empty DataFrame for {key}, skipping")
                result_dict[key] = df
                continue
            
            logger.info(f"Cleaning telemetry data for {key}")
            
            # Create a copy
            cleaned_df = df.copy()
            
            # Handle missing values
            cleaned_df = self.handle_missing_values(cleaned_df, strategy='linear')
            
            # Handle outliers in numeric columns
            numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
            cleaned_df = self.handle_outliers(cleaned_df, columns=numeric_cols, method='clip')
            
            result_dict[key] = cleaned_df
        
        return result_dict
    
    def prepare_modeling_dataset(self, race_df: pd.DataFrame, qualifying_df: pd.DataFrame, 
                                weather_df: pd.DataFrame, additional_dfs: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare a complete dataset for modeling, combining race, qualifying, and weather data.
        
        Args:
            race_df: DataFrame containing race results
            qualifying_df: DataFrame containing qualifying results
            weather_df: DataFrame containing weather data
            additional_dfs: Optional dictionary of additional DataFrames to include
            
        Returns:
            Combined and cleaned DataFrame ready for feature engineering
        """
        if race_df is None or race_df.empty:
            logger.error("Cannot prepare modeling dataset: race DataFrame is empty or None")
            return pd.DataFrame()
        
        logger.info("Preparing modeling dataset for F1 prediction")
        
        # Step 1: Standardize and clean the individual DataFrames
        race_clean = self.standardize_driver_names(race_df)
        race_clean = self.standardize_team_names(race_clean)
        if 'TrackName' in race_clean.columns:
            race_clean = self.standardize_circuit_names(race_clean, circuit_col='TrackName')
        
        qual_clean = None
        if qualifying_df is not None and not qualifying_df.empty:
            qual_clean = self.standardize_driver_names(qualifying_df)
            qual_clean = self.standardize_team_names(qual_clean)
            if 'TrackName' in qual_clean.columns:
                qual_clean = self.standardize_circuit_names(qual_clean, circuit_col='TrackName')
        
        weather_clean = None
        if weather_df is not None and not weather_df.empty:
            if 'circuit' in weather_df.columns:
                weather_clean = self.standardize_circuit_names(weather_df, circuit_col='circuit')
        
        # Step 2: Merge race and qualifying data
        if qual_clean is not None:
            combined_df = self.merge_race_qualifying_data(race_clean, qual_clean)
        else:
            combined_df = race_clean.copy()
        
        # Step 3: Merge with weather data
        if weather_clean is not None:
            combined_df = self.merge_with_weather_data(combined_df, weather_clean)
        
        # Step 4: Include any additional DataFrames if provided
        if additional_dfs is not None:
            for name, df in additional_dfs.items():
                if df is not None and not df.empty:
                    logger.info(f"Incorporating additional data from {name}")
                    
                    # Standardize and clean
                    add_df_clean = self.standardize_driver_names(df)
                    if 'Team' in add_df_clean.columns:
                        add_df_clean = self.standardize_team_names(add_df_clean)
                    if 'TrackName' in add_df_clean.columns:
                        add_df_clean = self.standardize_circuit_names(add_df_clean, circuit_col='TrackName')
                    elif 'Circuit' in add_df_clean.columns:
                        add_df_clean = self.standardize_circuit_names(add_df_clean, circuit_col='Circuit')
                    
                    # Determine merge columns
                    merge_cols = []
                    for col in ['Year', 'GrandPrix', 'Driver', 'TrackName', 'Circuit', 'Date']:
                        if col in combined_df.columns and col in add_df_clean.columns:
                            merge_cols.append(col)
                    
                    if not merge_cols:
                        logger.warning(f"No common columns found to merge with {name} data")
                        continue
                    
                    # Add prefix to avoid column name conflicts
                    prefix = name.lower() + '_'
                    rename_cols = {col: prefix + col for col in add_df_clean.columns if col not in merge_cols}
                    add_df_clean = add_df_clean.rename(columns=rename_cols)
                    
                    # Merge with the combined DataFrame
                    combined_df = pd.merge(combined_df, add_df_clean, on=merge_cols, how='left')
        
        # Step 5: Handle missing values in the combined dataset
        combined_df = self.handle_missing_values(combined_df, strategy='mean')
        
        # Step 6: Handle outliers in numeric columns
        numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
        combined_df = self.handle_outliers(combined_df, columns=numeric_cols, method='clip')
        
        logger.info(f"Final modeling dataset has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns")
        
        return combined_df
    
    def convert_lap_time_to_seconds(self, lap_time: Any) -> float:
        """
        Convert various lap time formats to seconds.
        
        Args:
            lap_time: Lap time in various formats (timedelta, string, etc.)
            
        Returns:
            Lap time in seconds as float
        """
        if pd.isna(lap_time):
            return np.nan
        
        # If already a number, return as is
        if isinstance(lap_time, (int, float)):
            return float(lap_time)
        
        # If timedelta, convert to seconds
        if isinstance(lap_time, pd.Timedelta):
            return lap_time.total_seconds()
        
        # If string, parse it
        if isinstance(lap_time, str):
            try:
                # Handle 'M:SS.mmm' format
                if ':' in lap_time and '.' in lap_time:
                    parts = lap_time.split(':')
                    minutes = float(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                    
                # Handle 'SS.mmm' format
                elif '.' in lap_time:
                    return float(lap_time)
                    
                # Handle other potential formats...
                else:
                    logger.warning(f"Unrecognized lap time format: {lap_time}")
                    return np.nan
                    
            except Exception as e:
                logger.error(f"Error parsing lap time '{lap_time}': {str(e)}")
                return np.nan
        
        # If none of the above worked
        logger.warning(f"Could not convert lap time of type {type(lap_time)}: {lap_time}")
        return np.nan
    
    def clean_race_weekend_data(self, practice_dfs: List[pd.DataFrame], 
                               qualifying_df: pd.DataFrame, 
                               race_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Clean and combine data for an entire race weekend (practice, qualifying, race).
        
        Args:
            practice_dfs: List of DataFrames containing practice session data
            qualifying_df: DataFrame containing qualifying data
            race_df: DataFrame containing race data
            
        Returns:
            Dictionary with cleaned DataFrames for each session type
        """
        result = {}
        
        # Clean practice data
        for i, practice_df in enumerate(practice_dfs):
            if practice_df is not None and not practice_df.empty:
                session_name = f"Practice{i+1}"
                logger.info(f"Cleaning {session_name} data")
                
                practice_clean = self.standardize_driver_names(practice_df)
                practice_clean = self.standardize_team_names(practice_clean)
                if 'TrackName' in practice_clean.columns:
                    practice_clean = self.standardize_circuit_names(practice_clean, circuit_col='TrackName')
                
                practice_clean = self.handle_missing_values(practice_clean, strategy='mean')
                result[session_name] = practice_clean
        
        # Clean qualifying data
        if qualifying_df is not None and not qualifying_df.empty:
            logger.info("Cleaning qualifying data")
            qual_clean = self.standardize_driver_names(qualifying_df)
            qual_clean = self.standardize_team_names(qual_clean)
            if 'TrackName' in qual_clean.columns:
                qual_clean = self.standardize_circuit_names(qual_clean, circuit_col='TrackName')
            
            qual_clean = self.handle_missing_values(qual_clean, strategy='mean')
            result['Qualifying'] = qual_clean
        
        # Clean race data
        if race_df is not None and not race_df.empty:
            logger.info("Cleaning race data")
            race_clean = self.standardize_driver_names(race_df)
            race_clean = self.standardize_team_names(race_clean)
            if 'TrackName' in race_clean.columns:
                race_clean = self.standardize_circuit_names(race_clean, circuit_col='TrackName')
            
            race_clean = self.handle_missing_values(race_clean, strategy='mean')
            result['Race'] = race_clean
        
        return result