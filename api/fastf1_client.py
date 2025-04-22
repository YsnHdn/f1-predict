"""
FastF1 API Client for F1 prediction project.
This module handles extraction of race data, qualifying data, and telemetry data from the FastF1 library.
It includes cache management for efficient data retrieval.
"""

import os
import logging
import pandas as pd
import fastf1
from fastf1 import plotting
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime

from api.cache.manager import CacheManager

# Configure logging
logger = logging.getLogger(__name__)

class FastF1Client:
    """
    Client for interacting with the FastF1 API to retrieve Formula 1 data.
    Handles caching and data formatting.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the FastF1 client with cache management.
        
        Args:
            cache_dir: Optional directory for cache. If None, default cache location is used.
        """
        # Configure FastF1 cache
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(cache_path)
        else:
            # Use default cache location (./.fastf1_cache)
            cache_path = Path('.fastf1_cache')
            cache_path.mkdir(parents=True, exist_ok=True)
            fastf1.Cache.enable_cache(cache_path)
        
        # Initialize the cache manager for our own caching layer
        self.cache_manager = CacheManager('fastf1')
        
        # Enable plotting settings
        plotting.setup_mpl(mpl_timedelta_support=True, color_scheme=None, misc_mpl_mods=True)
        
        logger.info("FastF1 client initialized with cache enabled")

    def get_session(self, year: int, gp_name: str, session_type: str) -> fastf1.core.Session:
        """
        Get a FastF1 session object.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S', 'SS')
        
        Returns:
            FastF1 Session object
        """
        cache_key = f"session_{year}_{gp_name}_{session_type}"
        cached_session = self.cache_manager.get(cache_key)
        
        if cached_session:
            logger.info(f"Using cached session for {year} {gp_name} {session_type}")
            return cached_session
        
        try:
            logger.info(f"Loading session data for {year} {gp_name} {session_type}")
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()
            
            # Store in cache
            self.cache_manager.set(cache_key, session)
            
            return session
        except Exception as e:
            logger.error(f"Error loading session data: {str(e)}")
            raise

    def get_race_calendar(self, year: int) -> pd.DataFrame:
        """
        Get the F1 race calendar for a specific year.
        
        Args:
            year: The season year
        
        Returns:
            DataFrame containing the race calendar
        """
        cache_key = f"calendar_{year}"
        cached_calendar = self.cache_manager.get(cache_key)
        
        if cached_calendar is not None:
            logger.info(f"Using cached calendar for {year}")
            return cached_calendar
        
        try:
            logger.info(f"Loading race calendar for {year}")
            calendar = fastf1.get_event_schedule(year)
            
            # Store in cache
            self.cache_manager.set(cache_key, calendar)
            
            return calendar
        except Exception as e:
            logger.error(f"Error loading race calendar: {str(e)}")
            raise

    def get_driver_standings(self, year: int, round_number: Optional[int] = None) -> pd.DataFrame:
        """
        Get driver standings for a specific year and optional round.
        
        Args:
            year: The season year
            round_number: Optional round number (if None, returns the latest standings)
        
        Returns:
            DataFrame containing driver standings
        """
        cache_key = f"driver_standings_{year}_{round_number}"
        cached_standings = self.cache_manager.get(cache_key)
        
        if cached_standings is not None:
            logger.info(f"Using cached driver standings for {year} round {round_number}")
            return cached_standings
        
        try:
            logger.info(f"Loading driver standings for {year} round {round_number}")
            
            # Get driver info first
            driver_info = self.get_driver_info(year)
            
            if driver_info is None or driver_info.empty:
                logger.error(f"Could not get driver info for {year}")
                return pd.DataFrame()
            
            # If we're looking at a future year or early in the season, 
            # create standings based on driver info only
            current_date = datetime.now()
            
            if year > current_date.year or (year == current_date.year and current_date.month < 3):
                # Create a basic standings DataFrame using driver info
                standings = pd.DataFrame({
                    'Position': range(1, len(driver_info) + 1),
                    'Driver': driver_info['Abbreviation'].values,
                    'Team': driver_info['TeamName'].values,
                    'Points': [0] * len(driver_info)  # No points yet
                })
                
                # Store in cache
                self.cache_manager.set(cache_key, standings)
                
                return standings
            
            # Récupérer le calendrier pour l'année
            calendar = self.get_race_calendar(year)
            
            # Récupérer tous les résultats de course jusqu'au round spécifié
            all_results = []
            for _, race in calendar.iterrows():
                race_name = race['EventName']
                race_round = race['RoundNumber'] if 'RoundNumber' in race else None
                race_date = pd.to_datetime(race['EventDate'])
                
                # Skip future races
                if race_date > current_date:
                    continue
                
                # Si round_number est spécifié, ne prendre que les courses jusqu'à ce round
                if round_number is not None and race_round is not None and race_round > round_number:
                    continue
                    
                try:
                    race_results = self.get_race_results(year, race_name)
                    if race_results is not None and not race_results.empty:
                        # Ensure consistent column names - standardize on 'Driver' column
                        if 'Driver' not in race_results.columns:
                            # Check available columns for driver identifier
                            if 'Abbreviation' in race_results.columns:
                                race_results['Driver'] = race_results['Abbreviation']
                            elif 'BroadcastName' in race_results.columns:
                                race_results['Driver'] = race_results['BroadcastName']
                            elif 'DriverNumber' in race_results.columns:
                                # Use driver info to map DriverNumber to Abbreviation
                                driver_mapping = dict(zip(driver_info['DriverNumber'], driver_info['Abbreviation']))
                                race_results['Driver'] = race_results['DriverNumber'].map(driver_mapping)
                            
                        # Also ensure Team column exists
                        if 'Team' not in race_results.columns and 'TeamName' in race_results.columns:
                            race_results['Team'] = race_results['TeamName']
                        
                        all_results.append(race_results)
                except Exception as e:
                    logger.warning(f"Could not get results for {race_name}: {str(e)}")
            
            # Compiler les standings à partir des résultats
            if all_results:
                # Concaténer tous les résultats
                combined_results = pd.concat(all_results, ignore_index=True)
                
                # Vérifier si la colonne Driver existe maintenant
                if 'Driver' not in combined_results.columns:
                    logger.error("Could not find or create Driver column in race results")
                    # Créer un classement basé sur les informations des pilotes
                    standings = pd.DataFrame({
                        'Position': range(1, len(driver_info) + 1),
                        'Driver': driver_info['Abbreviation'].values,
                        'Team': driver_info['TeamName'].values,
                        'Points': [0] * len(driver_info)
                    })
                else:
                    # Vérifier si Points est bien numérique
                    if 'Points' in combined_results.columns:
                        if not pd.api.types.is_numeric_dtype(combined_results['Points']):
                            combined_results['Points'] = pd.to_numeric(combined_results['Points'], errors='coerce').fillna(0)
                    else:
                        # Créer une colonne Points si elle n'existe pas
                        combined_results['Points'] = 0
                    
                    # Vérifier que Team existe
                    if 'Team' not in combined_results.columns:
                        combined_results['Team'] = 'Unknown'
                    
                    # Calculer les points totaux par pilote
                    standings = combined_results.groupby('Driver').agg({
                        'Points': 'sum',
                        'Team': 'first'  # Prendre la première équipe
                    }).reset_index()
                    
                    # Trier par points décroissants
                    standings = standings.sort_values('Points', ascending=False)
                    
                    # Ajouter le rang
                    standings['Position'] = range(1, len(standings) + 1)
                
                # Réorganiser les colonnes
                standings = standings[['Position', 'Driver', 'Team', 'Points']]
                
                # Store in cache
                self.cache_manager.set(cache_key, standings)
                
                return standings
            elif driver_info is not None:
                # Si aucun résultat mais pilotes disponibles, retourner les pilotes sans points
                standings = pd.DataFrame({
                    'Position': range(1, len(driver_info) + 1),
                    'Driver': driver_info['Abbreviation'].values,
                    'Team': driver_info['TeamName'].values,
                    'Points': [0] * len(driver_info)
                })
                
                # Store in cache
                self.cache_manager.set(cache_key, standings)
                
                return standings
            else:
                # Si aucun résultat, retourner un DataFrame vide avec les bonnes colonnes
                return pd.DataFrame(columns=['Position', 'Driver', 'Team', 'Points'])
        
        except Exception as e:
            logger.error(f"Error loading driver standings: {str(e)}")
            raise
    
    def get_constructor_standings(self, year: int, round_number: Optional[int] = None) -> pd.DataFrame:
        """
        Get constructor standings for a specific year and optional round.
        
        Args:
            year: The season year
            round_number: Optional round number (if None, returns the latest standings)
        
        Returns:
            DataFrame containing constructor standings
        """
        cache_key = f"constructor_standings_{year}_{round_number}"
        cached_standings = self.cache_manager.get(cache_key)
        
        if cached_standings is not None:
            logger.info(f"Using cached constructor standings for {year} round {round_number}")
            return cached_standings
        
        try:
            logger.info(f"Loading constructor standings for {year} round {round_number}")
            
            # Get driver info to extract teams
            driver_info = self.get_driver_info(year)
            
            if driver_info is None or driver_info.empty:
                logger.error(f"Could not get driver info for {year}")
                return pd.DataFrame()
            
            # Get unique teams from driver info
            teams = driver_info['TeamName'].unique()
            
            # If we're looking at a future year or early in the season,
            # create standings based on team info only
            current_date = datetime.now()
            
            if year > current_date.year or (year == current_date.year and current_date.month < 3):
                # Create a basic standings DataFrame using team info
                standings = pd.DataFrame({
                    'Position': range(1, len(teams) + 1),
                    'Team': teams,
                    'Points': [0] * len(teams)  # No points yet
                })
                
                # Store in cache
                self.cache_manager.set(cache_key, standings)
                
                return standings
            
            # Get driver standings
            driver_standings = self.get_driver_standings(year, round_number)
            
            if driver_standings is None or driver_standings.empty:
                logger.warning(f"No driver standings available for {year} round {round_number}")
                
                # Create empty constructor standings with teams from driver info
                if teams is not None and len(teams) > 0:
                    standings = pd.DataFrame({
                        'Position': range(1, len(teams) + 1),
                        'Team': teams,
                        'Points': [0] * len(teams)
                    })
                    
                    # Store in cache
                    self.cache_manager.set(cache_key, standings)
                    
                    return standings
                else:
                    return pd.DataFrame(columns=['Position', 'Team', 'Points'])
            
            # Aggregate driver points by team
            constructor_standings = driver_standings.groupby('Team')['Points'].sum().reset_index()
            
            # Sort by points in descending order
            constructor_standings = constructor_standings.sort_values('Points', ascending=False)
            
            # Add position
            constructor_standings['Position'] = range(1, len(constructor_standings) + 1)
            
            # Reorder columns
            constructor_standings = constructor_standings[['Position', 'Team', 'Points']]
            
            # Store in cache
            self.cache_manager.set(cache_key, constructor_standings)
            
            return constructor_standings
            
        except Exception as e:
            logger.error(f"Error loading constructor standings: {str(e)}")
            raise

    def get_race_results(self, year: int, gp_name: str) -> pd.DataFrame:
        """
        Get race results for a specific Grand Prix.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
        
        Returns:
            DataFrame containing race results
        """
        session = self.get_session(year, gp_name, 'R')
        results = session.results
        
        # Format the results DataFrame
        if results is not None:
            # Add session info
            results['Year'] = year
            results['GrandPrix'] = session.event['EventName']
            results['TrackName'] = session.event['OfficialEventName']
            results['Date'] = session.event['EventDate']
            
        return results

    def get_qualifying_results(self, year: int, gp_name: str) -> pd.DataFrame:
        """
        Get qualifying results for a specific Grand Prix.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
        
        Returns:
            DataFrame containing qualifying results
        """
        session = self.get_session(year, gp_name, 'Q')
        results = session.results
        
        # Format the results DataFrame
        if results is not None:
            # Add session info
            results['Year'] = year
            results['GrandPrix'] = session.event['EventName']
            results['TrackName'] = session.event['OfficialEventName']
            results['Date'] = session.event['EventDate']
            
        return results

    def get_sprint_results(self, year: int, gp_name: str) -> pd.DataFrame:
        """
        Get sprint race results for a specific Grand Prix.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
        
        Returns:
            DataFrame containing sprint results or None if no sprint
        """
        try:
            session = self.get_session(year, gp_name, 'S')
            results = session.results
            
            # Format the results DataFrame
            if results is not None:
                # Add session info
                results['Year'] = year
                results['GrandPrix'] = session.event['EventName']
                results['TrackName'] = session.event['OfficialEventName']
                results['Date'] = session.event['EventDate']
                
            return results
        except Exception as e:
            logger.warning(f"No sprint race for {year} {gp_name}: {str(e)}")
            return None

    def get_driver_telemetry(self, year: int, gp_name: str, session_type: str, 
                            driver: str, laps: Optional[Union[int, List[int]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get telemetry data for a specific driver in a session.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
            driver: Driver identifier (number or three-letter code)
            laps: Optional specific lap(s) to retrieve
        
        Returns:
            Dictionary with telemetry data frames (car_data, pos_data)
        """
        # Form a unique cache key
        lap_key = "_".join(map(str, laps)) if isinstance(laps, list) else str(laps)
        cache_key = f"telemetry_{year}_{gp_name}_{session_type}_{driver}_{lap_key}"
        
        cached_telemetry = self.cache_manager.get(cache_key)
        if cached_telemetry is not None:
            logger.info(f"Using cached telemetry for {driver} in {year} {gp_name} {session_type}")
            return cached_telemetry
            
        session = self.get_session(year, gp_name, session_type)
        
        # Get the laps for the driver
        driver_laps = session.laps.pick_driver(driver)
        
        if laps is not None:
            if isinstance(laps, int):
                driver_laps = driver_laps[driver_laps['LapNumber'] == laps]
            else:  # List of lap numbers
                driver_laps = driver_laps[driver_laps['LapNumber'].isin(laps)]
        
        # Initialize result dictionary
        telemetry_data = {
            'laps': driver_laps,
            'car_data': None,
            'pos_data': None
        }
        
        try:
            # Get car telemetry data
            car_data = pd.DataFrame()
            for _, lap in driver_laps.iterrows():
                lap_telemetry = lap.get_car_data()
                if lap_telemetry is not None:
                    lap_telemetry['LapNumber'] = lap['LapNumber']
                    car_data = pd.concat([car_data, lap_telemetry])
            
            telemetry_data['car_data'] = car_data
            
            # Get position data
            pos_data = pd.DataFrame()
            for _, lap in driver_laps.iterrows():
                lap_position = lap.get_pos_data()
                if lap_position is not None:
                    lap_position['LapNumber'] = lap['LapNumber']
                    pos_data = pd.concat([pos_data, lap_position])
            
            telemetry_data['pos_data'] = pos_data
            
            # Store in cache
            self.cache_manager.set(cache_key, telemetry_data)
            
            return telemetry_data
            
        except Exception as e:
            logger.error(f"Error retrieving telemetry data: {str(e)}")
            return telemetry_data

    def get_weather_data(self, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
        """
        Get weather data for a specific session.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
        
        Returns:
            DataFrame containing weather data
        """
        session = self.get_session(year, gp_name, session_type)
        
        try:
            weather_data = session.weather_data
            
            # Format the weather DataFrame
            if weather_data is not None:
                # Add session info
                weather_data['Year'] = year
                weather_data['GrandPrix'] = session.event['EventName']
                weather_data['Session'] = session_type
                
            return weather_data
        except Exception as e:
            logger.error(f"Error retrieving weather data: {str(e)}")
            return None

    def get_track_status_data(self, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
        """
        Get track status changes during a session (yellow flags, safety cars, etc.).
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
        
        Returns:
            DataFrame containing track status data
        """
        cache_key = f"track_status_{year}_{gp_name}_{session_type}"
        cached_status = self.cache_manager.get(cache_key)
        
        if cached_status is not None:
            logger.info(f"Using cached track status for {year} {gp_name} {session_type}")
            return cached_status
            
        session = self.get_session(year, gp_name, session_type)
        
        try:
            # Create a more usable track status DataFrame
            raw_status = session.race_control_messages
            
            if raw_status is None or raw_status.empty:
                return None
            
            # Add session info and meaningful status descriptions
            status_mapping = {
                1: "Track Clear",
                2: "Yellow Flag", 
                3: "Red Flag",
                4: "Safety Car",
                5: "Virtual Safety Car",
                6: "Virtual Safety Car Ending"
            }
            
            track_status = raw_status.copy()
            # Add status description if available
            if 'Status' in track_status.columns:
                track_status['StatusDesc'] = track_status['Status'].map(
                    lambda x: status_mapping.get(x, f"Unknown ({x})")
                )
            
            # Add session info
            track_status['Year'] = year
            track_status['GrandPrix'] = session.event['EventName']
            track_status['Session'] = session_type
            
            # Store in cache
            self.cache_manager.set(cache_key, track_status)
            
            return track_status
            
        except Exception as e:
            logger.error(f"Error retrieving track status data: {str(e)}")
            return None

    def get_driver_info(self, year: int) -> pd.DataFrame:
        """
        Get information about drivers for a specific season.
        
        Args:
            year: The season year
        
        Returns:
            DataFrame containing driver information
        """
        cache_key = f"drivers_{year}"
        cached_drivers = self.cache_manager.get(cache_key)
        
        if cached_drivers is not None:
            logger.info(f"Using cached driver info for {year}")
            return cached_drivers
            
        try:
            logger.info(f"Loading driver information for {year}")
            
            # Get the event schedule
            schedule = self.get_race_calendar(year)
            if schedule.empty:
                logger.error(f"No events found for {year}")
                return None
            
            # Find the first event that has already occurred
            current_date = datetime.now()
            event_to_use = None
            
            # First try to find a completed event
            for _, event in schedule.iterrows():
                event_date = pd.to_datetime(event['EventDate'])
                if event_date < current_date:
                    event_to_use = event['EventName']
                    break
            
            # If no completed event is found, use the first event in the calendar
            # This is useful for future seasons where no event has occurred yet
            if event_to_use is None and not schedule.empty:
                event_to_use = schedule.iloc[0]['EventName']
            
            if event_to_use is None:
                logger.error(f"Could not find any event for {year}")
                return None
                
            # Try to get a session from this event
            driver_info = None
            
            # Try different session types
            for session_type in ['R', 'Q', 'FP3', 'FP2', 'FP1']:
                try:
                    session = self.get_session(year, event_to_use, session_type)
                    
                    # Try to get driver info
                    driver_info = session.get_driver_info()
                    
                    if driver_info is not None and not driver_info.empty:
                        # We found driver info, break the loop
                        break
                except Exception as e:
                    logger.warning(f"Could not get {session_type} session for {event_to_use}: {str(e)}")
                    continue
            
            # If we still don't have driver info, create some
            if driver_info is None or driver_info.empty:
                # For 2025, create a current lineup of drivers
                if year == 2025:
                    logger.info("Creating 2025 driver lineup")
                    driver_info = pd.DataFrame({
                        'DriverNumber': [1, 11, 16, 55, 4, 81, 44, 63, 14, 18, 22, 3, 10, 31, 27, 20, 23, 21, 24, 77],
                        'BroadcastName': [
                            'VERSTAPPEN', 'PEREZ', 'LECLERC', 'SAINZ', 'NORRIS', 'PIASTRI', 
                            'HAMILTON', 'RUSSELL', 'ALONSO', 'STROLL', 'TSUNODA', 'RICCIARDO', 
                            'GASLY', 'OCON', 'HULKENBERG', 'MAGNUSSEN', 'ALBON', 'SARGEANT', 
                            'ZHOU', 'BOTTAS'
                        ],
                        'FullName': [
                            'Max VERSTAPPEN', 'Sergio PEREZ', 'Charles LECLERC', 'Carlos SAINZ', 
                            'Lando NORRIS', 'Oscar PIASTRI', 'Lewis HAMILTON', 'George RUSSELL', 
                            'Fernando ALONSO', 'Lance STROLL', 'Yuki TSUNODA', 'Daniel RICCIARDO', 
                            'Pierre GASLY', 'Esteban OCON', 'Nico HULKENBERG', 'Kevin MAGNUSSEN', 
                            'Alexander ALBON', 'Logan SARGEANT', 'ZHOU Guanyu', 'Valtteri BOTTAS'
                        ],
                        'Abbreviation': [
                            'VER', 'PER', 'LEC', 'SAI', 'NOR', 'PIA', 'HAM', 'RUS', 
                            'ALO', 'STR', 'TSU', 'RIC', 'GAS', 'OCO', 'HUL', 'MAG', 
                            'ALB', 'SAR', 'ZHO', 'BOT'
                        ],
                        'TeamName': [
                            'Red Bull Racing', 'Red Bull Racing', 'Ferrari', 'Ferrari', 
                            'McLaren', 'McLaren', 'Mercedes', 'Mercedes', 
                            'Aston Martin', 'Aston Martin', 'RB', 'RB', 
                            'Alpine', 'Alpine', 'Haas F1 Team', 'Haas F1 Team', 
                            'Williams', 'Williams', 'Stake F1 Team', 'Stake F1 Team'
                        ],
                        'TeamColor': [
                            '#3671C6', '#3671C6', '#F91536', '#F91536', 
                            '#FF8700', '#FF8700', '#27F4D2', '#27F4D2', 
                            '#358C75', '#358C75', '#6692FF', '#6692FF', 
                            '#2293D1', '#2293D1', '#B6BABD', '#B6BABD', 
                            '#37BEDD', '#37BEDD', '#C92D4B', '#C92D4B'
                        ]
                    })
                else:
                    logger.error(f"Could not get driver info for {year}")
                    return None
            
            # Add year information
            driver_info['Season'] = year
            
            # Store in cache
            self.cache_manager.set(cache_key, driver_info)
            
            return driver_info
            
        except Exception as e:
            logger.error(f"Error retrieving driver information: {str(e)}")
            return None

    def compare_drivers_lap_time(self, year: int, gp_name: str, session_type: str, 
                                drivers: List[str]) -> pd.DataFrame:
        """
        Compare lap times between multiple drivers.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
            drivers: List of driver identifiers
        
        Returns:
            DataFrame containing combined lap time data for comparison
        """
        session = self.get_session(year, gp_name, session_type)
        
        try:
            comparison_df = pd.DataFrame()
            
            for driver in drivers:
                driver_laps = session.laps.pick_driver(driver)
                
                if driver_laps.empty:
                    logger.warning(f"No lap data found for {driver}")
                    continue
                    
                # Add driver identifier
                driver_laps['Driver'] = driver
                
                # Append to comparison DataFrame
                comparison_df = pd.concat([comparison_df, driver_laps])
            
            # Add session info
            if not comparison_df.empty:
                comparison_df['Year'] = year
                comparison_df['GrandPrix'] = session.event['EventName']
                comparison_df['Session'] = session_type
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing driver lap times: {str(e)}")
            return None

    def get_session_lap_data(self, year: int, gp_name: str, session_type: str) -> pd.DataFrame:
        """
        Get complete lap data for all drivers in a session.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
        
        Returns:
            DataFrame containing lap data for all drivers
        """
        session = self.get_session(year, gp_name, session_type)
        
        try:
            lap_data = session.laps
            
            # Format the lap data DataFrame
            if lap_data is not None and not lap_data.empty:
                # Add session info
                lap_data['Year'] = year
                lap_data['GrandPrix'] = session.event['EventName']
                lap_data['TrackName'] = session.event['OfficialEventName']
                lap_data['Session'] = session_type
                
            return lap_data
            
        except Exception as e:
            logger.error(f"Error retrieving session lap data: {str(e)}")
            return None

    def get_fastest_laps(self, year: int, gp_name: str, session_type: str, n: int = 10) -> pd.DataFrame:
        """
        Get the fastest laps from a session.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
            session_type: Session type ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
            n: Number of fastest laps to return
        
        Returns:
            DataFrame containing the n fastest laps
        """
        session = self.get_session(year, gp_name, session_type)
        
        try:
            lap_data = session.laps
            
            if lap_data is None or lap_data.empty:
                return None
                
            # Filter out laps without a time
            valid_laps = lap_data.dropna(subset=['LapTime'])
            
            # Sort by lap time and get the n fastest
            fastest_laps = valid_laps.sort_values('LapTime').head(n)
            
            # Add session info
            fastest_laps['Year'] = year
            fastest_laps['GrandPrix'] = session.event['EventName']
            fastest_laps['TrackName'] = session.event['OfficialEventName']
            fastest_laps['Session'] = session_type
            
            return fastest_laps
            
        except Exception as e:
            logger.error(f"Error retrieving fastest laps: {str(e)}")
            return None

    def get_tyre_strategy(self, year: int, gp_name: str) -> pd.DataFrame:
        """
        Get tyre strategy information for a race.
        
        Args:
            year: The season year
            gp_name: Grand Prix name or round number
        
        Returns:
            DataFrame containing tyre strategy information
        """
        cache_key = f"tyre_strategy_{year}_{gp_name}"
        cached_strategy = self.cache_manager.get(cache_key)
        
        if cached_strategy is not None:
            logger.info(f"Using cached tyre strategy for {year} {gp_name}")
            return cached_strategy
            
        session = self.get_session(year, gp_name, 'R')
        
        try:
            lap_data = session.laps
            
            if lap_data is None or lap_data.empty:
                return None
                
            # Group by driver and stint to get strategy information
            tyre_stints = lap_data.groupby(['Driver', 'Stint']).agg({
                'Compound': 'first',
                'LapNumber': ['min', 'max', 'count'],
                'TyreLife': 'max'
            })
            
            # Flatten the multi-index columns
            tyre_stints.columns = ['_'.join(col).strip() for col in tyre_stints.columns.values]
            
            # Reset index for easier handling
            tyre_stints = tyre_stints.reset_index()
            
            # Add session info
            tyre_stints['Year'] = year
            tyre_stints['GrandPrix'] = session.event['EventName']
            
            # Store in cache
            self.cache_manager.set(cache_key, tyre_stints)
            
            return tyre_stints
            
        except Exception as e:
            logger.error(f"Error retrieving tyre strategy: {str(e)}")
            return None