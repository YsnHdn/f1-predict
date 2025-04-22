"""
Feature engineering module for F1 prediction project.
This module handles creation of predictive features from cleaned F1 and weather data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Configure logging
logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """
    Creates and transforms features for F1 race prediction models.
    Handles historical statistics, circuit-specific features, and weather impact.
    """
    
    def __init__(self, scale_features: bool = True, scaling_method: str = 'standard'):
        """
        Initialize the feature engineer.
        
        Args:
            scale_features: Whether to automatically scale numerical features
            scaling_method: Method for scaling ('standard', 'minmax', or None)
        """
        self.scale_features = scale_features
        self.scaling_method = scaling_method
        
        # Initialize scalers
        self.numerical_scaler = None
        if self.scale_features:
            if scaling_method == 'standard':
                self.numerical_scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.numerical_scaler = MinMaxScaler()
                
        # Initialize encoders
        try:
            # Essayer avec le nouveau paramètre (scikit-learn >= 1.2)
            self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # Fallback pour les anciennes versions (scikit-learn < 1.2)
            self.categorical_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        logger.info(f"Feature engineer initialized with scaling={scale_features}, method={scaling_method}")
    
    def create_grid_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to grid position and qualification performance.
        
        Args:
            df: DataFrame with race and qualifying data
            
        Returns:
            DataFrame with additional grid-related features
        """
        result_df = df.copy()
        
        # Check if we have grid position information
        grid_cols = ['GridPosition', 'Q_Position', 'grid', 'position']
        grid_col = next((col for col in grid_cols if col in result_df.columns), None)
        
        if grid_col is None:
            logger.warning("No grid position column found, skipping grid position features")
            return result_df
        
        logger.info(f"Creating grid position features based on {grid_col}")
        
        # Standardize column name
        if grid_col != 'GridPosition':
            result_df['GridPosition'] = result_df[grid_col]
        
        # Create features
        try:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(result_df['GridPosition']):
                result_df['GridPosition'] = pd.to_numeric(result_df['GridPosition'], errors='coerce')
            
            # Front row (positions 1-2)
            result_df['grid_front_row'] = (result_df['GridPosition'] <= 2).astype(int)
            
            # Top 3 positions
            result_df['grid_top3'] = (result_df['GridPosition'] <= 3).astype(int)
            
            # Top 5 positions
            result_df['grid_top5'] = (result_df['GridPosition'] <= 5).astype(int)
            
            # Top 10 positions
            result_df['grid_top10'] = (result_df['GridPosition'] <= 10).astype(int)
            
            # Grid position groups
            conditions = [
                (result_df['GridPosition'] == 1),
                (result_df['GridPosition'] > 1) & (result_df['GridPosition'] <= 3),
                (result_df['GridPosition'] > 3) & (result_df['GridPosition'] <= 6),
                (result_df['GridPosition'] > 6) & (result_df['GridPosition'] <= 10),
                (result_df['GridPosition'] > 10)
            ]
            
            values = ['pole', 'front', 'upper_midfield', 'midfield', 'back']
            result_df['grid_group'] = np.select(conditions, values, default='unknown')
            
        except Exception as e:
            logger.error(f"Error creating grid position features: {str(e)}")
        
        return result_df
    
    def create_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features related to qualifying performance.
        
        Args:
            df: DataFrame with qualifying data
            
        Returns:
            DataFrame with additional qualifying-related features
        """
        result_df = df.copy()
        
        # Check if we have qualifying time information
        q_time_cols = ['Q1', 'Q2', 'Q3', 'Q_Q1', 'Q_Q2', 'Q_Q3', 'Q_time', 'QualifyingTime']
        q_times_present = [col for col in q_time_cols if col in result_df.columns]
        
        if not q_times_present:
            logger.warning("No qualifying time columns found, skipping qualifying features")
            return result_df
        
        logger.info(f"Creating qualifying features based on {q_times_present}")
        
        # Create features
        try:
            # Convert qualifying times to seconds if not already numeric
            for col in q_times_present:
                if not pd.api.types.is_numeric_dtype(result_df[col]):
                    # Try to convert timedelta to seconds
                    try:
                        result_df[f'{col}_seconds'] = result_df[col].dt.total_seconds()
                    except:
                        # If not a timedelta, try to parse as a string time format
                        result_df[f'{col}_seconds'] = result_df[col].apply(lambda x: 
                            pd.to_datetime(x, format='%M:%S.%f').second + 
                            pd.to_datetime(x, format='%M:%S.%f').minute * 60 + 
                            pd.to_datetime(x, format='%M:%S.%f').microsecond / 1000000 
                            if isinstance(x, str) else np.nan
                        )
                else:
                    # Already numeric, just copy
                    result_df[f'{col}_seconds'] = result_df[col]
            
            # Qualifying sessions completed (how far driver made it: Q1, Q2, Q3)
            has_q1 = 'Q1_seconds' in result_df.columns or 'Q_Q1_seconds' in result_df.columns
            has_q2 = 'Q2_seconds' in result_df.columns or 'Q_Q2_seconds' in result_df.columns
            has_q3 = 'Q3_seconds' in result_df.columns or 'Q_Q3_seconds' in result_df.columns
            
            q1_col = 'Q1_seconds' if 'Q1_seconds' in result_df.columns else 'Q_Q1_seconds' if 'Q_Q1_seconds' in result_df.columns else None
            q2_col = 'Q2_seconds' if 'Q2_seconds' in result_df.columns else 'Q_Q2_seconds' if 'Q_Q2_seconds' in result_df.columns else None
            q3_col = 'Q3_seconds' if 'Q3_seconds' in result_df.columns else 'Q_Q3_seconds' if 'Q_Q3_seconds' in result_df.columns else None
            
            if has_q1 and has_q2 and has_q3:
                result_df['q_sessions_completed'] = np.where(result_df[q3_col].notna(), 3,
                                               np.where(result_df[q2_col].notna(), 2,
                                                      np.where(result_df[q1_col].notna(), 1, 0)))
            elif has_q1 and has_q2:
                result_df['q_sessions_completed'] = np.where(result_df[q2_col].notna(), 2,
                                               np.where(result_df[q1_col].notna(), 1, 0))
            elif has_q1:
                result_df['q_sessions_completed'] = np.where(result_df[q1_col].notna(), 1, 0)
            
            # Gap to pole position
            pole_cols = ['Q_pole_time', 'pole_time', 'Q3_seconds', 'Q_Q3_seconds', 'QualifyingTime_seconds']
            pole_col = next((col for col in pole_cols if col in result_df.columns), None)
            
            if pole_col is not None:
                # For each qualifying session, find the fastest time
                try:
                    # Group by relevant identifiers and find fastest qualifying time per session
                    groupby_cols = []
                    for col in ['Year', 'GrandPrix', 'TrackName', 'Circuit']:
                        if col in result_df.columns:
                            groupby_cols.append(col)
                    
                    if groupby_cols:
                        if 'Q3_seconds' in result_df.columns:
                            fastest_q3 = result_df.groupby(groupby_cols)['Q3_seconds'].min().reset_index()
                            fastest_q3.rename(columns={'Q3_seconds': 'fastest_q3'}, inplace=True)
                            result_df = pd.merge(result_df, fastest_q3, on=groupby_cols, how='left')
                            result_df['gap_to_pole_pct'] = (result_df['Q3_seconds'] - result_df['fastest_q3']) / result_df['fastest_q3'] * 100
                        elif 'Q_Q3_seconds' in result_df.columns:
                            fastest_q3 = result_df.groupby(groupby_cols)['Q_Q3_seconds'].min().reset_index()
                            fastest_q3.rename(columns={'Q_Q3_seconds': 'fastest_q3'}, inplace=True)
                            result_df = pd.merge(result_df, fastest_q3, on=groupby_cols, how='left')
                            result_df['gap_to_pole_pct'] = (result_df['Q_Q3_seconds'] - result_df['fastest_q3']) / result_df['fastest_q3'] * 100
                except Exception as e:
                    logger.error(f"Error calculating gap to pole: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error creating qualifying features: {str(e)}")
        
        return result_df
    
    def create_team_performance_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features related to team performance.
        
        Args:
            df: DataFrame with current race data
            historical_df: Optional DataFrame with historical performance data
            
        Returns:
            DataFrame with additional team-related features
        """
        result_df = df.copy()
    
        # Check if we have team information - élargi pour inclure TeamName
        team_cols = ['Team', 'team', 'Constructor', 'constructor', 'TeamName']
        team_col = next((col for col in team_cols if col in result_df.columns), None)
        
        if team_col is None:
            logger.warning("No team column found, skipping team performance features")
            return result_df
        
        logger.info(f"Creating team performance features based on {team_col}")
        
        # Standardize column name
        if team_col != 'Team':
            result_df['Team'] = result_df[team_col]
        
        # Create features from current data
        try:
            # If we have qualifying/grid data, compute average team qualifying position
            if 'GridPosition' in result_df.columns:
                # Group by relevant identifiers
                groupby_cols = ['Team']
                for col in ['Year', 'GrandPrix', 'TrackName', 'Circuit']:
                    if col in result_df.columns:
                        groupby_cols.append(col)
                
                # Calculate team's average grid position for this race
                team_grid = result_df.groupby(groupby_cols)['GridPosition'].mean().reset_index()
                team_grid.rename(columns={'GridPosition': 'team_avg_grid'}, inplace=True)
                result_df = pd.merge(result_df, team_grid, on=groupby_cols, how='left')
                
                # Calculate driver's position relative to team average
                result_df['grid_vs_teammate'] = result_df['GridPosition'] - result_df['team_avg_grid']
            
            # If we have historical data, create more detailed team performance features
            if historical_df is not None:
                logger.info("Using historical data for team performance features")
                
                # Standardize team column in historical data
                historical_team_col = next((col for col in team_cols if col in historical_df.columns), None)
                if historical_team_col is not None and historical_team_col != 'Team':
                    historical_df['Team'] = historical_df[historical_team_col]
                
                # Calculate team's historical performance
                if 'Points' in historical_df.columns:
                    # Average points per race
                    team_points = historical_df.groupby('Team')['Points'].mean().reset_index()
                    team_points.rename(columns={'Points': 'team_avg_points'}, inplace=True)
                    result_df = pd.merge(result_df, team_points, on='Team', how='left')
                
                # Team's finish reliability
                if 'Status' in historical_df.columns:
                    # Calculate finish rate
                    team_finishes = historical_df.groupby('Team').apply(
                        lambda x: (x['Status'].str.contains('Finished', case=False) | 
                                  ~x['Status'].str.contains('DNF|DNS|DSQ', case=False, regex=True)).mean()
                    ).reset_index(name='team_finish_rate')
                    
                    result_df = pd.merge(result_df, team_finishes, on='Team', how='left')
                
                # Team's performance on specific track types
                if 'TrackName' in historical_df.columns and 'TrackName' in result_df.columns:
                    # Define track types (simplified)
                    street_circuits = ['monaco', 'baku', 'singapore', 'jeddah', 'las_vegas']
                    high_speed_circuits = ['monza', 'spa', 'silverstone', 'suzuka', 'spielberg', 'barcelona']
                    
                    # Add track type info to historical data
                    historical_df['is_street_circuit'] = historical_df['TrackName'].str.lower().isin(street_circuits)
                    historical_df['is_high_speed_circuit'] = historical_df['TrackName'].str.lower().isin(high_speed_circuits)
                    
                    # Calculate team performance by track type
                    if 'Position' in historical_df.columns:
                        # Street circuits
                        street_perf = historical_df[historical_df['is_street_circuit']].groupby('Team')['Position'].mean().reset_index()
                        street_perf.rename(columns={'Position': 'team_street_circuit_avg_pos'}, inplace=True)
                        result_df = pd.merge(result_df, street_perf, on='Team', how='left')
                        
                        # High-speed circuits
                        speed_perf = historical_df[historical_df['is_high_speed_circuit']].groupby('Team')['Position'].mean().reset_index()
                        speed_perf.rename(columns={'Position': 'team_high_speed_circuit_avg_pos'}, inplace=True)
                        result_df = pd.merge(result_df, speed_perf, on='Team', how='left')
                    
                    # Add current track type to result
                    result_df['is_street_circuit'] = result_df['TrackName'].str.lower().isin(street_circuits)
                    result_df['is_high_speed_circuit'] = result_df['TrackName'].str.lower().isin(high_speed_circuits)
                
        except Exception as e:
            logger.error(f"Error creating team performance features: {str(e)}")
        
        return result_df
    
    def create_driver_performance_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features related to driver performance.
        
        Args:
            df: DataFrame with current race data
            historical_df: Optional DataFrame with historical performance data
            
        Returns:
            DataFrame with additional driver-related features
        """
        result_df = df.copy()
    
        # Check if we have driver information - élargi pour inclure Abbreviation
        driver_cols = ['Driver', 'driver', 'Abbreviation', 'BroadcastName', 'DriverNumber']
        driver_col = next((col for col in driver_cols if col in result_df.columns), None)
        
        if driver_col is None:
            logger.warning("No driver column found, skipping driver performance features")
            return result_df
        
        logger.info(f"Creating driver performance features based on {driver_col}")
        
        # Standardize column name
        if driver_col != 'Driver':
            result_df['Driver'] = result_df[driver_col]
        
        # Create features
        try:
            # If we have historical data, create detailed driver performance features
            if historical_df is not None:
                logger.info("Using historical data for driver performance features")
                
                # Standardize driver column in historical data
                historical_driver_col = next((col for col in driver_cols if col in historical_df.columns), None)
                if historical_driver_col is not None and historical_driver_col != 'Driver':
                    historical_df['Driver'] = historical_df[historical_driver_col]
                
                # Ensure Position column exists in historical data
                if 'Position' not in historical_df.columns:
                    # Try alternative columns for position
                    position_cols = ['ClassifiedPosition', 'FinishingPosition', 'position']
                    pos_col = next((col for col in position_cols if col in historical_df.columns), None)
                    if pos_col:
                        historical_df['Position'] = pd.to_numeric(historical_df[pos_col], errors='coerce')
                    elif 'Status' in historical_df.columns:
                        # Extract position from Status if it contains position info
                        historical_df['Position'] = historical_df['Status'].str.extract(r'Finished (\d+)').astype(float)
                
                # Calculate driver's historical performance
                if 'Points' in historical_df.columns:
                    # Average points per race
                    driver_points = historical_df.groupby('Driver')['Points'].mean().reset_index()
                    driver_points.rename(columns={'Points': 'driver_avg_points'}, inplace=True)
                    result_df = pd.merge(result_df, driver_points, on='Driver', how='left')
                
                # Driver's start vs finish position (ability to gain/lose positions)
                if 'GridPosition' in historical_df.columns and 'Position' in historical_df.columns:
                    # Calculate average positions gained/lost
                    historical_df['positions_gained'] = historical_df['GridPosition'] - historical_df['Position']
                    driver_gains = historical_df.groupby('Driver')['positions_gained'].mean().reset_index()
                    driver_gains.rename(columns={'positions_gained': 'driver_avg_positions_gained'}, inplace=True)
                    result_df = pd.merge(result_df, driver_gains, on='Driver', how='left')
                
                # Driver's finish reliability
                if 'Status' in historical_df.columns:
                    # Calculate finish rate
                    driver_finishes = historical_df.groupby('Driver').apply(
                        lambda x: (x['Status'].str.contains('Finished', case=False) | 
                                ~x['Status'].str.contains('DNF|DNS|DSQ', case=False, regex=True)).mean()
                    ).reset_index(name='driver_finish_rate')
                    
                    result_df = pd.merge(result_df, driver_finishes, on='Driver', how='left')
                    
                    # Driver's performance on specific track types
                    if 'TrackName' in historical_df.columns and 'TrackName' in result_df.columns:
                        # Define track types (simplified)
                        street_circuits = ['monaco', 'baku', 'singapore', 'jeddah', 'las_vegas']
                        high_speed_circuits = ['monza', 'spa', 'silverstone', 'suzuka', 'spielberg', 'barcelona']
                        
                        # Add track type info to historical data
                        historical_df['is_street_circuit'] = historical_df['TrackName'].str.lower().isin(street_circuits)
                        historical_df['is_high_speed_circuit'] = historical_df['TrackName'].str.lower().isin(high_speed_circuits)
                        
                        # Calculate driver performance by track type
                        if 'Position' in historical_df.columns:
                            # Street circuits
                            street_perf = historical_df[historical_df['is_street_circuit']].groupby('Driver')['Position'].mean().reset_index()
                            street_perf.rename(columns={'Position': 'driver_street_circuit_avg_pos'}, inplace=True)
                            result_df = pd.merge(result_df, street_perf, on='Driver', how='left')
                            
                            # High-speed circuits
                            speed_perf = historical_df[historical_df['is_high_speed_circuit']].groupby('Driver')['Position'].mean().reset_index()
                            speed_perf.rename(columns={'Position': 'driver_high_speed_circuit_avg_pos'}, inplace=True)
                            result_df = pd.merge(result_df, speed_perf, on='Driver', how='left')
                
                # Driver's performance on current track
                if 'TrackName' in historical_df.columns and 'TrackName' in result_df.columns:
                    # Get all unique current track names
                    current_tracks = result_df['TrackName'].unique()
                    
                    for track in current_tracks:
                        # Filter historical data for this track
                        track_data = historical_df[historical_df['TrackName'] == track]
                        
                        if not track_data.empty and 'Position' in track_data.columns:
                            # Calculate average position for each driver at this track
                            track_perf = track_data.groupby('Driver')['Position'].mean().reset_index()
                            track_perf.rename(columns={'Position': f'driver_{track.lower()}_avg_pos'}, inplace=True)
                            
                            # Merge with current data for this track only
                            track_mask = result_df['TrackName'] == track
                            result_df = pd.merge(
                                result_df, 
                                track_perf, 
                                on='Driver', 
                                how='left'
                            )
                
                # Driver's recent form (last 3 races)
                if 'Date' in historical_df.columns and 'Date' in result_df.columns:
                    # Ensure Date columns are datetime
                    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
                    result_df['Date'] = pd.to_datetime(result_df['Date'])
                    
                    # For each race in the current data
                    for idx, row in result_df.iterrows():
                        current_date = row['Date']
                        driver = row['Driver']
                        
                        # Get last 3 races for this driver before current date
                        last_races = historical_df[
                            (historical_df['Driver'] == driver) & 
                            (historical_df['Date'] < current_date)
                        ].sort_values('Date', ascending=False).head(3)
                        
                        # Calculate average recent position if we have data
                        if not last_races.empty and 'Position' in last_races.columns:
                            result_df.loc[idx, 'driver_last3_avg_pos'] = last_races['Position'].mean()
                            
                            # Also calculate trend (improving or declining)
                            if len(last_races) >= 2:
                                positions = last_races.sort_values('Date', ascending=True)['Position'].values
                                # Positive value means declining (higher position numbers = worse)
                                # Negative value means improving
                                trend = np.polyfit(range(len(positions)), positions, 1)[0]
                                result_df.loc[idx, 'driver_form_trend'] = trend
        
        except Exception as e:
            logger.error(f"Error creating driver performance features: {str(e)}")
        
        return result_df
                          
    def create_circuit_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features related to circuit characteristics.
        
        Args:
            df: DataFrame with current race data
            historical_df: Optional DataFrame with historical performance data
            
        Returns:
            DataFrame with additional circuit-related features
        """
        result_df = df.copy()
    
        # Check if we have circuit information - élargi pour inclure OfficialEventName
        circuit_cols = ['TrackName', 'Circuit', 'track', 'circuit', 'OfficialEventName', 'EventName']
        circuit_col = next((col for col in circuit_cols if col in result_df.columns), None)
        
        if circuit_col is None:
            logger.warning("No circuit column found, skipping circuit features")
            return result_df
        
        logger.info(f"Creating circuit features based on {circuit_col}")
        
        # Standardize column name
        if circuit_col != 'TrackName':
            result_df['TrackName'] = result_df[circuit_col]
        
        # Create features
        try:
            # Circuit type features
            # Define circuit types based on characteristics
            street_circuits = ['monaco', 'baku', 'singapore', 'jeddah', 'las_vegas']
            high_speed_circuits = ['monza', 'spa', 'silverstone', 'suzuka', 'spielberg', 'barcelona']
            technical_circuits = ['hungaroring', 'zandvoort', 'monaco', 'rodriguez']
            
            # Add circuit type indicators
            result_df['circuit_street'] = result_df['TrackName'].str.lower().isin(street_circuits).astype(int)
            result_df['circuit_high_speed'] = result_df['TrackName'].str.lower().isin(high_speed_circuits).astype(int)
            result_df['circuit_technical'] = result_df['TrackName'].str.lower().isin(technical_circuits).astype(int)
            
            # If we have historical data, create more detailed circuit features
            if historical_df is not None and circuit_col in historical_df.columns:
                logger.info("Using historical data for circuit features")
                
                # Standardize circuit column in historical data if needed
                if circuit_col != 'TrackName' and circuit_col in historical_df.columns:
                    historical_df['TrackName'] = historical_df[circuit_col]
                
                # Overtaking difficulty (based on positions gained/lost)
                if 'GridPosition' in historical_df.columns and 'Position' in historical_df.columns:
                    # Calculate average positions gained/lost per circuit
                    historical_df['positions_gained'] = historical_df['GridPosition'] - historical_df['Position']
                    circuit_gains = historical_df.groupby('TrackName')['positions_gained'].mean().reset_index()
                    circuit_gains.rename(columns={'positions_gained': 'circuit_avg_positions_gained'}, inplace=True)
                    result_df = pd.merge(result_df, circuit_gains, on='TrackName', how='left')
                    
                    # Overtaking difficulty index (lower value = harder to overtake)
                    circuit_gains['overtaking_difficulty'] = -circuit_gains['circuit_avg_positions_gained']
                    circuit_gains['overtaking_difficulty'] = (circuit_gains['overtaking_difficulty'] - 
                                                           circuit_gains['overtaking_difficulty'].min()) / (
                                                           circuit_gains['overtaking_difficulty'].max() - 
                                                           circuit_gains['overtaking_difficulty'].min())
                    
                    circuit_difficulty = circuit_gains[['TrackName', 'overtaking_difficulty']]
                    result_df = pd.merge(result_df, circuit_difficulty, on='TrackName', how='left')
                
                # Safety car likelihood
                if 'Status' in historical_df.columns:
                    # Calculate percentage of races with safety car mentions
                    safety_car_terms = ['Safety Car', 'safety car', 'VSC', 'virtual safety car']
                    historical_df['had_safety_car'] = historical_df['Status'].str.contains('|'.join(safety_car_terms), case=False, regex=True).astype(int)
                    
                    safety_car_rate = historical_df.groupby('TrackName')['had_safety_car'].mean().reset_index()
                    safety_car_rate.rename(columns={'had_safety_car': 'circuit_safety_car_rate'}, inplace=True)
                    result_df = pd.merge(result_df, safety_car_rate, on='TrackName', how='left')
        
        except Exception as e:
            logger.error(f"Error creating circuit features: {str(e)}")
        
        return result_df
    
    def create_weather_impact_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features related to weather impact on performance.
        
        Args:
            df: DataFrame with current race and weather data
            historical_df: Optional DataFrame with historical performance data
            
        Returns:
            DataFrame with additional weather impact features
        """
        result_df = df.copy()
        
        # Check if we have weather information
        weather_cols = [col for col in result_df.columns if col.startswith('weather_')]
        
        if not weather_cols:
            logger.warning("No weather columns found, skipping weather impact features")
            return result_df
        
        logger.info(f"Creating weather impact features based on {len(weather_cols)} weather variables")
        
        # Create features
        try:
            # Weather condition category
            if 'weather_racing_condition' in result_df.columns:
                # Create indicator variables for different conditions
                result_df['weather_is_dry'] = (result_df['weather_racing_condition'] == 'dry').astype(int)
                result_df['weather_is_damp'] = (result_df['weather_racing_condition'] == 'damp').astype(int)
                result_df['weather_is_wet'] = (result_df['weather_racing_condition'] == 'wet').astype(int)
                result_df['weather_is_very_wet'] = (result_df['weather_racing_condition'] == 'very_wet').astype(int)
                
                # Simplified condition (binary dry/wet)
                result_df['weather_is_any_wet'] = ((result_df['weather_racing_condition'] == 'damp') |
                                                (result_df['weather_racing_condition'] == 'wet') |
                                                (result_df['weather_racing_condition'] == 'very_wet')).astype(int)
            
            # Wind impact (high wind makes cars harder to handle)
            if 'weather_wind_speed_ms' in result_df.columns:
                # Define high wind threshold (8 m/s is approximately 29 km/h)
                wind_threshold = 8
                result_df['weather_high_wind'] = (result_df['weather_wind_speed_ms'] >= wind_threshold).astype(int)
            
            # Temperature impact (affects tire behavior)
            if 'weather_temp_celsius' in result_df.columns:
                # Temperature categories
                result_df['weather_temp_cold'] = (result_df['weather_temp_celsius'] < 15).astype(int)
                result_df['weather_temp_mild'] = ((result_df['weather_temp_celsius'] >= 15) & 
                                               (result_df['weather_temp_celsius'] < 25)).astype(int)
                result_df['weather_temp_hot'] = (result_df['weather_temp_celsius'] >= 25).astype(int)
            
            # If we have historical data, create driver/team weather performance features
            if historical_df is not None and 'weather_racing_condition' in historical_df.columns:
                logger.info("Using historical data for weather impact features")
                
                # Driver performance in different conditions
                if 'Driver' in historical_df.columns and 'Driver' in result_df.columns and 'Position' in historical_df.columns:
                    # Performance in wet conditions
                    wet_mask = historical_df['weather_is_any_wet'] == 1
                    if wet_mask.any():
                        wet_perf = historical_df[wet_mask].groupby('Driver')['Position'].mean().reset_index()
                        wet_perf.rename(columns={'Position': 'driver_wet_condition_avg_pos'}, inplace=True)
                        result_df = pd.merge(result_df, wet_perf, on='Driver', how='left')
                    
                    # Performance in dry conditions
                    dry_mask = historical_df['weather_is_dry'] == 1
                    if dry_mask.any():
                        dry_perf = historical_df[dry_mask].groupby('Driver')['Position'].mean().reset_index()
                        dry_perf.rename(columns={'Position': 'driver_dry_condition_avg_pos'}, inplace=True)
                        result_df = pd.merge(result_df, dry_perf, on='Driver', how='left')
                    
                    # Calculate wet vs dry advantage
                    if 'driver_wet_condition_avg_pos' in result_df.columns and 'driver_dry_condition_avg_pos' in result_df.columns:
                        result_df['driver_wet_advantage'] = result_df['driver_dry_condition_avg_pos'] - result_df['driver_wet_condition_avg_pos']
                
                # Team performance in different conditions
                if 'Team' in historical_df.columns and 'Team' in result_df.columns and 'Position' in historical_df.columns:
                    # Performance in wet conditions
                    wet_mask = historical_df['weather_is_any_wet'] == 1
                    if wet_mask.any():
                        team_wet_perf = historical_df[wet_mask].groupby('Team')['Position'].mean().reset_index()
                        team_wet_perf.rename(columns={'Position': 'team_wet_condition_avg_pos'}, inplace=True)
                        result_df = pd.merge(result_df, team_wet_perf, on='Team', how='left')
                    
                    # Performance in dry conditions
                    dry_mask = historical_df['weather_is_dry'] == 1
                    if dry_mask.any():
                        team_dry_perf = historical_df[dry_mask].groupby('Team')['Position'].mean().reset_index()
                        team_dry_perf.rename(columns={'Position': 'team_dry_condition_avg_pos'}, inplace=True)
                        result_df = pd.merge(result_df, team_dry_perf, on='Team', how='left')
                    
                    # Calculate wet vs dry advantage
                    if 'team_wet_condition_avg_pos' in result_df.columns and 'team_dry_condition_avg_pos' in result_df.columns:
                        result_df['team_wet_advantage'] = result_df['team_dry_condition_avg_pos'] - result_df['team_wet_condition_avg_pos']
        
        except Exception as e:
            logger.error(f"Error creating weather impact features: {str(e)}")
        
        return result_df
    
    def create_race_strategy_features(self, df: pd.DataFrame, tyre_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features related to race strategy and tire usage.
        
        Args:
            df: DataFrame with race data
            tyre_data: Optional DataFrame with tire strategy data
            
        Returns:
            DataFrame with additional strategy-related features
        """
        result_df = df.copy()
        
        # Create features from available race data
        try:
            if 'TrackName' in result_df.columns:
                # Circuit types often require different strategies
                high_degradation_circuits = ['barcelona', 'spielberg', 'bahrain', 'shanghai']
                low_degradation_circuits = ['monaco', 'zandvoort', 'hungaroring']
                
                result_df['circuit_high_degradation'] = result_df['TrackName'].str.lower().isin(high_degradation_circuits).astype(int)
                result_df['circuit_low_degradation'] = result_df['TrackName'].str.lower().isin(low_degradation_circuits).astype(int)
            
            # If we have tire data, create more detailed strategy features
            if tyre_data is not None and not tyre_data.empty:
                logger.info("Using tire data for strategy features")
                
                # Ensure we have required columns
                required_cols = ['Driver', 'Stint', 'Compound']
                if not all(col in tyre_data.columns for col in required_cols):
                    logger.warning("Tire data missing required columns, skipping tire strategy features")
                    return result_df
                
                # Standardize driver names
                from preprocessing.data_cleaning import F1DataCleaner
                cleaner = F1DataCleaner()
                tyre_data = cleaner.standardize_driver_names(tyre_data)
                
                # Number of pitstops per driver
                if 'Stint' in tyre_data.columns:
                    pit_stops = tyre_data.groupby('Driver')['Stint'].max().reset_index()
                    pit_stops['PitStops'] = pit_stops['Stint']
                    pit_stops = pit_stops[['Driver', 'PitStops']]
                    result_df = pd.merge(result_df, pit_stops, on='Driver', how='left')
                
                # Tire compound usage
                if 'Compound' in tyre_data.columns:
                    # Count laps on each compound
                    compound_usage = tyre_data.groupby(['Driver', 'Compound']).size().reset_index(name='laps_on_compound')
                    
                    # Pivot to get one column per compound
                    compound_pivot = compound_usage.pivot(index='Driver', columns='Compound', values='laps_on_compound').reset_index()
                    compound_pivot.columns.name = None
                    
                    # Rename columns to avoid conflicts
                    compound_cols = {col: f'laps_on_{col.lower()}' for col in compound_pivot.columns if col != 'Driver'}
                    compound_pivot = compound_pivot.rename(columns=compound_cols)
                    
                    # Merge with result DataFrame
                    result_df = pd.merge(result_df, compound_pivot, on='Driver', how='left')
                    
                    # Fill NaN values with 0 for compound columns
                    for col in compound_cols.values():
                        if col in result_df.columns:
                            result_df[col] = result_df[col].fillna(0)
                    
                    # Calculate preferred compound (the one with most laps)
                    compound_cols_list = list(compound_cols.values())
                    if compound_cols_list:
                        result_df['preferred_compound'] = result_df[compound_cols_list].idxmax(axis=1)
                        result_df['preferred_compound'] = result_df['preferred_compound'].str.replace('laps_on_', '')
        
        except Exception as e:
            logger.error(f"Error creating race strategy features: {str(e)}")
        
        return result_df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df: DataFrame with categorical features
            categorical_cols: List of categorical columns to encode (None means auto-detect)
            
        Returns:
            DataFrame with encoded categorical features
        """
        result_df = df.copy()
        
        # If no categorical columns specified, auto-detect
        if categorical_cols is None:
            categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Filter out high-cardinality columns and ID-like columns
            excluded_cols = ['Driver', 'Team', 'TrackName', 'GrandPrix', 'Status']
            categorical_cols = [col for col in categorical_cols if col not in excluded_cols]
        
        if not categorical_cols:
            logger.info("No categorical columns to encode")
            return result_df
        
        logger.info(f"Encoding {len(categorical_cols)} categorical features: {categorical_cols}")
        
        try:
            # Use one-hot encoding
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            for col in categorical_cols:
                # Skip columns with too many unique values or missing values
                if col not in result_df.columns:
                    continue
                    
                unique_count = result_df[col].nunique()
                if unique_count > 10:  # Skip if too many categories
                    logger.warning(f"Skipping encoding for column {col} with {unique_count} unique values")
                    continue
                
                # Reshape for encoder
                X = result_df[col].values.reshape(-1, 1)
                
                # Fit and transform
                encoded = encoder.fit_transform(X)
                
                # Get feature names
                categories = encoder.categories_[0]
                feature_names = [f"{col}_{cat}" for cat in categories]
                
                # Create a DataFrame with encoded values
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=result_df.index)
                
                # Add encoded columns to result
                result_df = pd.concat([result_df, encoded_df], axis=1)
                
                # Drop original column
                result_df = result_df.drop(col, axis=1)
        
        except Exception as e:
            logger.error(f"Error encoding categorical features: {str(e)}")
        
        return result_df
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features using the configured scaler.
        
        Args:
            df: DataFrame with numerical features
            numerical_cols: List of numerical columns to scale (None means auto-detect)
            
        Returns:
            DataFrame with scaled numerical features
        """
        if not self.scale_features or self.numerical_scaler is None:
            logger.info("Feature scaling is disabled")
            return df
        
        result_df = df.copy()
        
        # If no numerical columns specified, auto-detect
        if numerical_cols is None:
            numerical_cols = result_df.select_dtypes(include=['number']).columns.tolist()
            
            # Filter out ID-like columns and target variables
            excluded_cols = ['Year', 'DriverNumber', 'Number', 'Position', 'Points']
            numerical_cols = [col for col in numerical_cols if col not in excluded_cols]
        
        if not numerical_cols:
            logger.info("No numerical columns to scale")
            return result_df
        
        logger.info(f"Scaling {len(numerical_cols)} numerical features using {self.scaling_method}")
        
        try:
            # Extract columns to scale
            X = result_df[numerical_cols].values
            
            # Handle missing values before scaling
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            
            # Scale features
            X_scaled = self.numerical_scaler.fit_transform(X)
            
            # Update result DataFrame
            result_df[numerical_cols] = X_scaled
        
        except Exception as e:
            logger.error(f"Error scaling numerical features: {str(e)}")
        
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features that combine multiple factors.
        
        Args:
            df: DataFrame with existing features
            
        Returns:
            DataFrame with additional interaction features
        """
        result_df = df.copy()
        
        logger.info("Creating interaction features")
        
        try:
            # Grid position × Circuit overtaking difficulty
            if 'GridPosition' in result_df.columns and 'overtaking_difficulty' in result_df.columns:
                result_df['grid_overtaking_interaction'] = result_df['GridPosition'] * result_df['overtaking_difficulty']
            
            # Weather × Driver wet performance advantage
            if 'weather_is_any_wet' in result_df.columns and 'driver_wet_advantage' in result_df.columns:
                result_df['wet_driver_advantage_interaction'] = result_df['weather_is_any_wet'] * result_df['driver_wet_advantage']
            
            # Team performance × Circuit type
            if 'team_high_speed_circuit_avg_pos' in result_df.columns and 'circuit_high_speed' in result_df.columns:
                result_df['team_highspeed_interaction'] = result_df['circuit_high_speed'] * result_df['team_high_speed_circuit_avg_pos']
            
            if 'team_street_circuit_avg_pos' in result_df.columns and 'circuit_street' in result_df.columns:
                result_df['team_street_interaction'] = result_df['circuit_street'] * result_df['team_street_circuit_avg_pos']
        
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
        
        return result_df
    
    def create_all_features(self, df: pd.DataFrame, historical_df: Optional[pd.DataFrame] = None, 
                          tyre_data: Optional[pd.DataFrame] = None, encode_categorical: bool = True) -> pd.DataFrame:
        """
        Create all features and apply scaling/encoding as configured.
        
        Args:
            df: Main DataFrame with race and qualifying data
            historical_df: Optional DataFrame with historical performance data
            tyre_data: Optional DataFrame with tire strategy data
            encode_categorical: Whether to encode categorical features
            
        Returns:
            DataFrame with all engineered features
        """
        if df is None or df.empty:
            logger.error("Cannot create features: input DataFrame is empty or None")
            return pd.DataFrame()
        
        logger.info("Creating all features for F1 prediction model")
        
        # Start with cleaned data
        result_df = df.copy()
        
        # Apply all feature engineering steps
        result_df = self.create_grid_position_features(result_df)
        result_df = self.create_qualifying_features(result_df)
        result_df = self.create_team_performance_features(result_df, historical_df)
        result_df = self.create_driver_performance_features(result_df, historical_df)
        result_df = self.create_circuit_features(result_df, historical_df)
        result_df = self.create_weather_impact_features(result_df, historical_df)
        result_df = self.create_race_strategy_features(result_df, tyre_data)
        result_df = self.create_interaction_features(result_df)
        
        # Encode categorical features if requested
        if encode_categorical:
            result_df = self.encode_categorical_features(result_df)
        
        # Scale numerical features if configured
        if self.scale_features and self.numerical_scaler is not None:
            result_df = self.scale_numerical_features(result_df)
        
        logger.info(f"Feature engineering complete: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
        
        return result_df