"""
F1 Data Manager - Handles data collection, updates and organization
"""

import os
import sys
import logging
import shutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import required modules
from api.fastf1_client import FastF1Client
from agents.data_collector import DataCollectorAgent
from agents.weather_monitor import WeatherMonitorAgent
from preprocessing.feature_engineering import F1FeatureEngineer
from preprocessing.data_cleaning import F1DataCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("f1_data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("F1DataManager")

class F1DataManager:
    """
    Manages F1 data collection, updates, and organization for the prediction model.
    Handles both initial setup and incremental updates.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Base directory for storing data
        """
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.features_dir = os.path.join(self.processed_dir, "features")
        self.predictions_dir = os.path.join(data_dir, "predictions")
        self.training_dir = os.path.join(data_dir, "training")
        
        # Initialize agents and clients
        self.fastf1_client = FastF1Client()
        self.data_collector = DataCollectorAgent()
        self.weather_monitor = WeatherMonitorAgent()
        self.feature_engineer = F1FeatureEngineer(scale_features=False)
        self.data_cleaner = F1DataCleaner()
        
        # Create necessary directories
        self._setup_directories()
        
        # Track metadata
        self.data_status = self._load_data_status()
        
        logger.info(f"F1DataManager initialized with data directory: {data_dir}")
    
    def _setup_directories(self):
        """Create necessary directory structure if it doesn't exist."""
        directories = [
            self.raw_dir,
            self.processed_dir,
            self.features_dir,
            self.predictions_dir,
            self.training_dir,
            os.path.join(self.raw_dir, "historical"),
            os.path.join(self.raw_dir, "practice"),
            os.path.join(self.raw_dir, "qualifying"),
            os.path.join(self.raw_dir, "races"),
            os.path.join(self.raw_dir, "telemetry"),
            os.path.join(self.raw_dir, "weather"),
            os.path.join(self.raw_dir, "tests"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        # Create FastF1 cache directory
        cache_dir = Path('.fastf1_cache')
        if not cache_dir.exists():
            os.makedirs(cache_dir)
            logger.info(f"Created FastF1 cache directory: {cache_dir}")
    
    def _load_data_status(self) -> Dict:
        """
        Load data status from metadata file or initialize if not exists.
        The status file tracks which races have been processed and when.
        
        Returns:
            Dictionary with data status information
        """
        status_file = os.path.join(self.processed_dir, "data_status.json")
        
        if os.path.exists(status_file):
            try:
                import json
                with open(status_file, 'r') as f:
                    status = json.load(f)
                logger.info(f"Loaded existing data status from {status_file}")
                return status
            except Exception as e:
                logger.error(f"Error loading data status: {str(e)}")
                
        # Initialize new status
        status = {
            "last_update": datetime.now().isoformat(),
            "races_processed": {},
            "historical_data": {
                "collected": False,
                "last_update": None,
                "years": []
            },
            "current_year": datetime.now().year,
            "next_race": None,
        }
        
        # Save initial status
        self._save_data_status(status)
        
        return status
    
    def _save_data_status(self, status: Dict = None):
        """
        Save data status to metadata file.
        
        Args:
            status: Status dictionary to save (uses self.data_status if None)
        """
        if status is None:
            status = self.data_status
            
        # Update timestamp
        status["last_update"] = datetime.now().isoformat()
        
        # Convert pandas Timestamp objects to ISO format strings
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            elif str(type(obj)).find('Timestamp') >= 0:  # Check for pandas Timestamp
                return obj.isoformat()
            else:
                return obj
        
        # Convert timestamp objects
        sanitized_status = convert_timestamps(status)
        
        # Save to file
        status_file = os.path.join(self.processed_dir, "data_status.json")
        try:
            import json
            with open(status_file, 'w') as f:
                json.dump(sanitized_status, f, indent=2, default=str)  # Use default=str as fallback for any other non-serializable objects
            logger.info(f"Saved data status to {status_file}")
        except Exception as e:
            logger.error(f"Error saving data status: {str(e)}")
    
    
    def get_next_race(self) -> Dict[str, Any]:
        """
        Get information about the next upcoming race.
        
        Returns:
            Dictionary with next race information
        """
        logger.info("Getting next race information")
        
        try:
            # Check if we already have cached next race info that's still valid
            if self.data_status["next_race"] is not None:
                next_race = self.data_status["next_race"]
                # Parse the date
                race_date = datetime.fromisoformat(next_race["date"].replace('Z', '+00:00') if isinstance(next_race["date"], str) else next_race["date"])
                
                # If the cached race is still in the future, use it
                if race_date > datetime.now():
                    logger.info(f"Using cached next race: {next_race['name']} on {next_race['date']}")
                    return next_race
            
            # Get current schedule
            year = datetime.now().year
            schedule = self.fastf1_client.get_race_calendar(year)
            
            # Filter future races
            future_races = schedule[pd.to_datetime(schedule['EventDate']) > datetime.now()]
            
            if future_races.empty:
                logger.warning(f"No upcoming races found for {year}")
                return None
            
            # Get the next race
            next_race_row = future_races.iloc[0]
            next_race = {
                'name': next_race_row['EventName'],
                'circuit': next_race_row['Location'],  # Use Location instead of OfficialEventName
                'date': next_race_row['EventDate'],
                'round': int(next_race_row['RoundNumber']) if 'RoundNumber' in next_race_row else None
            }
            
            # Cache the result
            self.data_status["next_race"] = next_race
            self._save_data_status()
            
            logger.info(f"Next race: {next_race['name']} at {next_race['circuit']} on {next_race['date']}")
            return next_race
            
        except Exception as e:
            logger.error(f"Error getting next race: {str(e)}")
            return None
        
    def collect_initial_data(self, years_back: int = 5) -> bool:
        """
        Perform initial data collection for historical races.
        This should only be run once when setting up the prediction system.
        
        Args:
            years_back: Number of years of historical data to collect
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting initial data collection (years_back={years_back})")
        
        # Check if initial collection has already been done
        if self.data_status["historical_data"]["collected"]:
            logger.warning("Initial data collection has already been performed")
            return True
        
        try:
            # Get current year
            current_year = datetime.now().year
            
            # Collect historical data for previous years
            years_collected = []
            for year_offset in range(1, years_back + 1):
                year = current_year - year_offset
                
                logger.info(f"Collecting data for {year} season")
                
                # Get schedule for the year
                schedule = self.fastf1_client.get_race_calendar(year)
                
                if schedule is None or schedule.empty:
                    logger.warning(f"No race schedule found for {year}")
                    continue
                
                # Process each race in that year
                for _, race in schedule.iterrows():
                    gp_name = race['EventName']
                    
                    logger.info(f"Collecting data for {year} {gp_name}")
                    
                    # Use the data collector agent
                    context = {
                        'year': year,
                        'gp_name': gp_name,
                        'historical_years': 0  # Just this specific race
                    }
                    
                    try:
                        result = self.data_collector.run(context)
                        
                        # Track this race as processed
                        race_key = f"{year}_{gp_name}"
                        self.data_status["races_processed"][race_key] = {
                            "year": year,
                            "name": gp_name,
                            "circuit": race['OfficialEventName'],
                            "date": race['EventDate'],
                            "collected": datetime.now().isoformat(),
                            "data_paths": result.get("data_paths", {})
                        }
                        
                        # Save status after each race
                        self._save_data_status()
                        
                    except Exception as e:
                        logger.error(f"Error collecting data for {year} {gp_name}: {str(e)}")
                
                # Mark year as collected
                years_collected.append(year)
            
            # Update status
            self.data_status["historical_data"]["collected"] = True
            self.data_status["historical_data"]["last_update"] = datetime.now().isoformat()
            self.data_status["historical_data"]["years"] = years_collected
            self._save_data_status()
            
            # Create the combined historical dataset
            self._combine_historical_data()
            
            logger.info(f"Initial data collection completed for years: {years_collected}")
            return True
            
        except Exception as e:
            logger.error(f"Error during initial data collection: {str(e)}")
            return False
    
    def _combine_historical_data(self) -> str:
        """
        Combine all historical race data into a single dataset.
        This includes ALL previously completed races in the current season.
        
        Returns:
            Path to the combined dataset file
        """
        logger.info("Combining historical race data")
        
        try:
            # Get the current year
            current_year = datetime.now().year
            
            # Get all completed races from the current season
            current_calendar = self.fastf1_client.get_race_calendar(current_year)
            completed_races = []
            
            if current_calendar is not None and not current_calendar.empty:
                # Filter for races that have already happened
                completed_races_rows = current_calendar[pd.to_datetime(current_calendar['EventDate']) < datetime.now()]
                
                for _, race in completed_races_rows.iterrows():
                    gp_name = race['EventName']
                    try:
                        # Get race results
                        race_results = self.fastf1_client.get_race_results(current_year, gp_name)
                        if race_results is not None and not race_results.empty:
                            logger.info(f"Adding {current_year} {gp_name} to historical data")
                            completed_races.append(race_results)
                            
                            # Save individual race results if not already saved
                            race_key = f"{current_year}_{gp_name}"
                            if race_key not in self.data_status["races_processed"]:
                                file_path = os.path.join(
                                    self.raw_dir, 
                                    "historical", 
                                    f"{current_year}_{gp_name.replace(' ', '_')}_R_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                )
                                race_results.to_csv(file_path, index=False)
                                
                                # Add to processed races
                                self.data_status["races_processed"][race_key] = {
                                    "year": current_year,
                                    "name": gp_name,
                                    "circuit": race['OfficialEventName'],
                                    "date": race['EventDate'],
                                    "collected": datetime.now().isoformat(),
                                    "data_paths": {f"race_{gp_name}": file_path}
                                }
                    except Exception as e:
                        logger.warning(f"Error collecting data for {current_year} {gp_name}: {str(e)}")
            
            # Get races from previous seasons from the processed list
            previous_race_files = []
            for race_key, race_info in self.data_status["races_processed"].items():
                data_paths = race_info.get("data_paths", {})
                
                # Look for race results or any relevant data
                for key, path in data_paths.items():
                    if any(term in key.lower() for term in ["race", "result", "historical"]):
                        if os.path.exists(path):
                            previous_race_files.append(path)
            
            # Combine all data
            all_race_files = [race_df for race_df in completed_races]  # Start with current season races
            
            # Add previous seasons' races from files
            for file in previous_race_files:
                try:
                    df = pd.read_csv(file)
                    all_race_files.append(df)
                except Exception as e:
                    logger.warning(f"Error reading file {file}: {str(e)}")
            
            if not all_race_files:
                logger.warning("No race data found to combine")
                return None
            
            logger.info(f"Combining {len(all_race_files)} race datasets")
            
            # Combine all dataframes
            combined_df = pd.concat(all_race_files, ignore_index=True)
            
            # Clean the data
            combined_df = self.data_cleaner.standardize_driver_names(combined_df)
            combined_df = self.data_cleaner.standardize_team_names(combined_df)
            
            if 'TrackName' in combined_df.columns:
                combined_df = self.data_cleaner.standardize_circuit_names(combined_df)
            
            # Clean data types
            if hasattr(self.data_cleaner, 'clean_data_types'):
                combined_df = self.data_cleaner.clean_data_types(combined_df)
            
            # Save the combined dataset
            timestamp = datetime.now().strftime("%Y%m%d")
            combined_file = os.path.join(self.processed_dir, f"historical_combined_{timestamp}.csv")
            combined_df.to_csv(combined_file, index=False)
            
            logger.info(f"Combined historical data saved to {combined_file}")
            
            # Update status with the file location
            self.data_status["historical_data"]["combined_file"] = combined_file
            self._save_data_status()
            
            return combined_file
            
        except Exception as e:
            logger.error(f"Error combining historical data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    
    """
Fix for the data_manager.py to resolve the circuit variable reference issue
"""

    def update_data_for_race(self, race_name: str = None, race_date: str = None) -> bool:
        """
        Update data for a specific race or the next upcoming race.
        
        Args:
            race_name: Name of the race to update (None for next race)
            race_date: Date of the race (None for next race)
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Updating data for race: {race_name or 'next race'}")
        
        try:
            # Get next race info if not specified
            if race_name is None or race_date is None:
                next_race = self.get_next_race()
                if next_race is None:
                    logger.error("Could not determine next race")
                    return False
                
                race_name = next_race['name']
                race_date = next_race['date']
                circuit = next_race['circuit']
            else:
                # If race is specified but circuit is not, try to find it
                circuit = None
                try:
                    # Try to get race information from FastF1
                    year = datetime.now().year
                    schedule = self.fastf1_client.get_race_calendar(year)
                    race_match = schedule[schedule['EventName'] == race_name]
                    
                    if not race_match.empty:
                        circuit = race_match.iloc[0]['Location']  # Use Location instead of OfficialEventName
                    else:
                        # Fallback: use race_name as circuit identifier
                        circuit = race_name
                        logger.warning(f"Could not find circuit info for {race_name}, using race name as identifier")
                except Exception as e:
                    # Fallback: use race_name as circuit identifier
                    circuit = race_name
                    logger.warning(f"Error finding circuit for {race_name}: {str(e)}, using race name as identifier")
            
            # Determine the year
            if isinstance(race_date, str):
                # Handle ISO format dates by converting to YYYY-MM-DD format
                if 'T' in race_date:
                    race_date = race_date.split('T')[0]  # Extract just the YYYY-MM-DD part
                
                year = datetime.strptime(race_date, '%Y-%m-%d').year
            else:
                year = race_date.year
                # Convert datetime to string in YYYY-MM-DD format
                race_date = race_date.strftime('%Y-%m-%d')
            
            # Create context for data collection
            context = {
                'year': year,
                'gp_name': race_name,
                'historical_years': 3
            }
            
            # Run data collection
            data_result = self.data_collector.run(context)
            
            if not data_result:
                logger.error(f"Failed to collect data for {race_name}")
                return False
            
            # Get weather data - ensuring race_date is in the correct format YYYY-MM-DD
            weather_context = {
                'circuit': circuit,
                'race_date': race_date,  # Now properly formatted as YYYY-MM-DD
                'days_range': 3
            }
            
            logger.info(f"Getting weather data for circuit: {circuit}, date: {race_date}")
            weather_result = self.weather_monitor.run(weather_context)
            
            # Track this race as processed
            race_key = f"{year}_{race_name}"
            self.data_status["races_processed"][race_key] = {
                "year": year,
                "name": race_name,
                "circuit": circuit,
                "date": race_date,
                "collected": datetime.now().isoformat(),
                "data_paths": data_result.get("data_paths", {}),
                "weather_paths": weather_result.get("data_paths", {}) if weather_result else {}
            }
            
            # Save status
            self._save_data_status()
            
            # Update combined historical data
            self._combine_historical_data()
            
            logger.info(f"Successfully updated data for {race_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating data for race: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    
    def generate_features_for_race(self, race_name: str = None, race_date: str = None) -> str:
        """
        Generate features for a specific race or the next upcoming race.
        
        Args:
            race_name: Name of the race (None for next race)
            race_date: Date of the race (None for next race)
            
        Returns:
            Path to the generated feature file or None if failed
        """
        logger.info(f"Generating features for race: {race_name or 'next race'}")
        
        try:
            # Get next race info if not specified
            if race_name is None or race_date is None:
                next_race = self.get_next_race()
                if next_race is None:
                    logger.error("Could not determine next race")
                    return None
                
                race_name = next_race['name']
                race_date = next_race['date']
                circuit = next_race['circuit']
            else:
                # Get circuit if not available
                circuit = race_name  # Fallback if we don't have the official name
            
            # Determine the year
            if isinstance(race_date, str):
                year = datetime.fromisoformat(race_date).year
            else:
                year = race_date.year
            
            # Check if we have data for this race
            race_key = f"{year}_{race_name}"
            if race_key not in self.data_status["races_processed"]:
                logger.warning(f"No data found for {race_name}. Collecting data first.")
                success = self.update_data_for_race(race_name, race_date)
                if not success:
                    logger.error(f"Failed to collect data for {race_name}")
                    return None
            
            # Load historical data
            historical_file = self.data_status["historical_data"].get("combined_file")
            if historical_file is None or not os.path.exists(historical_file):
                logger.warning("No combined historical data found. Creating it.")
                historical_file = self._combine_historical_data()
                if historical_file is None:
                    logger.error("Failed to create combined historical data")
                    return None
            
            historical_df = pd.read_csv(historical_file)
            
            # Create a template for the upcoming race
            # Get current driver standings
            standings_file = None
            race_info = self.data_status["races_processed"].get(race_key, {})
            data_paths = race_info.get("data_paths", {})
            
            for key, path in data_paths.items():
                if "driver_standings" in key.lower() and os.path.exists(path):
                    standings_file = path
                    break
            
            if standings_file is None:
                logger.warning("No driver standings found for feature generation")
                return None
            
            # Load driver standings
            driver_standings = pd.read_csv(standings_file)
            
            # Get driver info
            driver_info_file = None
            for key, path in data_paths.items():
                if "driver" in key.lower() and "info" in key.lower() and os.path.exists(path):
                    driver_info_file = path
                    break
            
            if driver_info_file is None:
                logger.warning("No driver info found for feature generation")
                # We'll proceed without it, using just standings
            else:
                driver_info = pd.read_csv(driver_info_file)
                # Merge driver info with standings
                if 'Driver' in driver_standings.columns and 'Abbreviation' in driver_info.columns:
                    driver_standings = pd.merge(
                        driver_standings,
                        driver_info[['Abbreviation', 'TeamName', 'DriverNumber']],
                        left_on='Driver',
                        right_on='Abbreviation',
                        how='left'
                    )
            
            # Create race template
            if isinstance(race_date, str):
                race_date_obj = datetime.fromisoformat(race_date)
            else:
                race_date_obj = race_date
                
            race_template = pd.DataFrame({
                'Driver': driver_standings['Driver'].values,
                'Team': driver_standings['Team'].values if 'Team' in driver_standings.columns 
                       else driver_standings['TeamName'].values if 'TeamName' in driver_standings.columns
                       else ['Unknown'] * len(driver_standings),
                'TrackName': [circuit] * len(driver_standings),
                'Year': [year] * len(driver_standings),
                'GrandPrix': [race_name] * len(driver_standings),
                'Date': [race_date_obj.strftime('%Y-%m-%d')] * len(driver_standings)
            })
            
            # Standardize names
            race_template = self.data_cleaner.standardize_driver_names(race_template)
            race_template = self.data_cleaner.standardize_team_names(race_template)
            race_template = self.data_cleaner.standardize_circuit_names(race_template)
            
            # Add weather data if available
            weather_data = None
            weather_paths = race_info.get("weather_paths", {})
            
            for key, path in weather_paths.items():
                if "race_day" in key.lower() and os.path.exists(path):
                    weather_data = pd.read_csv(path)
                    break
            
            if weather_data is not None:
                logger.info("Adding weather data to race template")
                
                # Extract race conditions
                race_conditions = {}
                for col in weather_data.columns:
                    if col.startswith('weather_') or col in ['temp_celsius', 'rain_mm', 'wind_speed_ms', 'racing_condition']:
                        # Take the mean for numerical values, most common for categoricals
                        if pd.api.types.is_numeric_dtype(weather_data[col]):
                            race_conditions[col] = weather_data[col].mean()
                        else:
                            race_conditions[col] = weather_data[col].mode()[0]
                
                # Add each condition to race template
                for key, value in race_conditions.items():
                    race_template[key] = value
            
            # Generate features
            logger.info("Generating features using feature engineering")
            features_df = self.feature_engineer.create_all_features(
                race_template, 
                historical_df=historical_df,
                encode_categorical=False
            )
            
            # Save feature file
            timestamp = datetime.now().strftime("%Y%m%d")
            race_name_clean = race_name.replace(' ', '_').lower()
            feature_file = os.path.join(self.features_dir, f"{timestamp}_{race_name_clean}_features.csv")
            features_df.to_csv(feature_file, index=False)
            
            # Update status with feature file
            if race_key in self.data_status["races_processed"]:
                self.data_status["races_processed"][race_key]["feature_file"] = feature_file
                self._save_data_status()
            
            logger.info(f"Features generated and saved to {feature_file}")
            return feature_file
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def check_features_readiness(self, feature_file: str) -> Tuple[bool, List[str]]:
        """
        Check if a feature file has all required features for model training.
        
        Args:
            feature_file: Path to the feature file to check
            
        Returns:
            Tuple with (is_ready, missing_features)
        """
        logger.info(f"Checking feature readiness for {feature_file}")
        
        try:
            # Load feature file
            features_df = pd.read_csv(feature_file)
            
            # Get required features
            from models.initial_model import F1InitialModel
            model = F1InitialModel()
            required_features = model.features
            
            # Check for missing features
            missing_features = [f for f in required_features if f not in features_df.columns]
            
            # Ready if no missing features or only a few non-critical ones
            is_ready = len(missing_features) <= 3  # Allow some flexibility
            
            if is_ready:
                logger.info(f"Feature file is ready for training with {len(required_features) - len(missing_features)}/{len(required_features)} features")
            else:
                logger.warning(f"Feature file is missing {len(missing_features)} required features")
                
            return is_ready, missing_features
            
        except Exception as e:
            logger.error(f"Error checking feature readiness: {str(e)}")
            return False, ["Error checking features"]
    
    def update_and_prepare_for_next_race(self) -> Dict[str, Any]:
        """
        Complete workflow to update data and prepare features for the next race.
        
        Returns:
            Dictionary with workflow results
        """
        logger.info("Starting complete workflow for next race preparation")
        
        results = {
            "success": False,
            "next_race": None,
            "data_updated": False,
            "features_generated": False,
            "feature_file": None,
            "features_ready": False,
            "missing_features": []
        }
        
        try:
            # 1. Get next race info
            next_race = self.get_next_race()
            if next_race is None:
                logger.error("Could not determine next race")
                return results
            
            results["next_race"] = next_race
            
            # 2. Update data for the race
            data_updated = self.update_data_for_race(
                next_race['name'], 
                next_race['date']
            )
            
            results["data_updated"] = data_updated
            
            if not data_updated:
                logger.error("Failed to update data for next race")
                return results
            
            # 3. Generate features
            feature_file = self.generate_features_for_race(
                next_race['name'],
                next_race['date']
            )
            
            results["feature_file"] = feature_file
            results["features_generated"] = feature_file is not None
            
            if feature_file is None:
                logger.error("Failed to generate features for next race")
                return results
            
            # 4. Check feature readiness
            is_ready, missing_features = self.check_features_readiness(feature_file)
            
            results["features_ready"] = is_ready
            results["missing_features"] = missing_features
            
            # Overall success
            results["success"] = data_updated and feature_file is not None
            
            logger.info(f"Next race preparation complete: {results['success']}")
            return results
            
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            return results


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="F1 Data Management Tool")
    
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Base directory for data storage")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize parser
    init_parser = subparsers.add_parser("init", help="Initialize data collection")
    init_parser.add_argument("--years", type=int, default=5,
                           help="Number of years of historical data to collect")
    
    # Update parser
    update_parser = subparsers.add_parser("update", help="Update data for next race")
    update_parser.add_argument("--race", type=str, default=None,
                             help="Specific race name to update (default: next race)")
    
    # Features parser
    features_parser = subparsers.add_parser("features", help="Generate features for next race")
    features_parser.add_argument("--race", type=str, default=None,
                               help="Specific race name to generate features for (default: next race)")
    
    # Full workflow parser
    subparsers.add_parser("workflow", help="Run full update and preparation workflow")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create data manager
    data_manager = F1DataManager(data_dir=args.data_dir)
    
    # Execute command
    if args.command == "init":
        print(f"Initializing data collection (years_back={args.years})...")
        success = data_manager.collect_initial_data(years_back=args.years)
        print(f"Initialization {'successful' if success else 'failed'}")
        
    elif args.command == "update":
        print(f"Updating data for {'next race' if args.race is None else args.race}...")
        success = data_manager.update_data_for_race(race_name=args.race)
        print(f"Update {'successful' if success else 'failed'}")
        
    elif args.command == "features":
        print(f"Generating features for {'next race' if args.race is None else args.race}...")
        feature_file = data_manager.generate_features_for_race(race_name=args.race)
        if feature_file:
            print(f"Features generated successfully: {feature_file}")
            
            # Check feature readiness
            is_ready, missing_features = data_manager.check_features_readiness(feature_file)
            
            if is_ready:
                print("Features are READY for model training!")
            else:
                print(f"Features are MISSING {len(missing_features)} required elements:")
                for feature in missing_features:
                    print(f"  - {feature}")
        else:
            print("Feature generation failed")
            
    elif args.command == "workflow":
        print("Running full preparation workflow...")
        results = data_manager.update_and_prepare_for_next_race()
        
        if results["success"]:
            print(f"Workflow completed successfully for {results['next_race']['name']}")
            print(f"Features generated: {results['feature_file']}")
            print(f"Features ready: {results['features_ready']}")
            
            if not results["features_ready"]:
                print(f"Missing features: {', '.join(results['missing_features'])}")
        else:
            print("Workflow failed")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main()