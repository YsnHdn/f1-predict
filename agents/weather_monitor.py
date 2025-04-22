"""
Weather monitor agent for F1 prediction project.
This agent is responsible for collecting and monitoring weather data for F1 races.
"""

import logging
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from crewai import Task

from agents.base_agent import F1BaseAgent
from agents.utils.logging import AgentLogger
from api.visualcrossing_client import VisualCrossingClient  # Importation du nouveau client

# Configure logging
logger = logging.getLogger(__name__)

class WeatherMonitorAgent(F1BaseAgent):
    """
    Agent responsible for monitoring weather conditions for F1 races.
    Collects forecast data, historical weather data, and alerts for upcoming races.
    Updates weather data as race day approaches for more accurate predictions.
    """
    
    def __init__(self, data_dir: str = "data/raw/weather"):
        """
        Initialize the weather monitor agent.
        
        Args:
            data_dir: Directory to store weather data
        """
        super().__init__(
            name="Weather Monitor",
            description="Monitors and analyzes weather conditions for Formula 1 races",
            goal="Provide accurate and timely weather information for race prediction"
        )
        
        self.data_dir = data_dir
        self.weather_client = VisualCrossingClient()  # Utiliser le nouveau client
        self.agent_logger = AgentLogger(agent_name="WeatherMonitor")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Create subdirectories
        for subdir in ['forecast', 'historical', 'alerts']:
            path = os.path.join(data_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        self.agent_logger.info(f"Initialized WeatherMonitorAgent with data directory: {data_dir}")
        
    def get_backstory(self) -> str:
            """
            Get the agent's backstory for CrewAI.
            
            Returns:
                String containing the agent's backstory
            """
            return (
                "I am a meteorological expert specializing in weather analysis for motorsport events. "
                "With years of experience studying how weather conditions affect racing performance, "
                "I understand the critical impact that temperature, precipitation, wind, and track "
                "conditions have on Formula 1 races. I continuously monitor weather patterns around "
                "F1 circuits worldwide, providing highly accurate forecasts and historical context "
                "to help teams and analysts make informed decisions."
            )
        
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Execute weather monitoring and data collection.
            
            Args:
                context: Context with information needed for weather monitoring
                    Expected keys:
                    - circuit: Circuit name/identifier
                    - race_date: Date of the race (string in YYYY-MM-DD format or datetime)
                    - days_range: Number of days before and after race to collect weather data
                    - update_frequency: How frequently to update forecasts (in hours)
                    
            Returns:
                Dictionary with collected weather data paths and current conditions
            """
            if context is None:
                context = {}
            
            # Extract parameters from context
            circuit = context.get('circuit', None)
            race_date = context.get('race_date', None)
            days_range = context.get('days_range', 3)
            update_frequency = context.get('update_frequency', 6)  # hours
            
            # Validate inputs
            if circuit is None:
                self.agent_logger.error("No circuit specified for weather monitoring")
                raise ValueError("Circuit name is required for weather monitoring")
            
            # Convert race_date string to datetime if needed
            if isinstance(race_date, str):
                try:
                    race_date = datetime.strptime(race_date, '%Y-%m-%d')
                except ValueError:
                    self.agent_logger.error(f"Invalid race date format: {race_date}")
                    raise ValueError("Race date must be in YYYY-MM-DD format")
            
            self.agent_logger.info(f"Starting weather monitoring for: Circuit={circuit}, Race date={race_date}")
            
            # Calculate date range
            if race_date:
                start_date = race_date - timedelta(days=days_range)
                end_date = race_date + timedelta(days=days_range)
            else:
                # If no race date provided, use current date
                current_date = datetime.now()
                start_date = current_date - timedelta(days=days_range)
                end_date = current_date + timedelta(days=days_range)
                race_date = current_date  # Use current date as race date
            
            # Store results
            results = {
                'circuit': circuit,
                'race_date': race_date.strftime('%Y-%m-%d'),
                'data_paths': {},
                'current_conditions': None,
                'forecast': None,
                'race_day_forecast': None,
                'historical_data': None,
                'weather_alerts': None,
                'weather_impact': None
            }
            
            try:
                # 1. Get current weather conditions
                self.agent_logger.task_start("Fetching current weather conditions")
                current_weather = self.weather_client.get_current_weather(circuit)
                
                if current_weather is not None and not current_weather.empty:
                    # Save current weather data
                    current_weather_path = self._save_data_to_file(
                        current_weather, f"{circuit}_current",
                        subdir='forecast'
                    )
                    
                    results['current_conditions'] = current_weather_path
                    results['data_paths']['current_conditions'] = current_weather_path
                    
                    # Also store the actual data for immediate use
                    current_weather_dict = current_weather.to_dict(orient='records')
                    if current_weather_dict:
                        results['current_weather_data'] = current_weather_dict[0]
                
                self.agent_logger.task_complete("Fetching current weather conditions")
                
                # 2. Get weather forecast for the race weekend
                self.agent_logger.task_start("Fetching weather forecast")
                forecast = self.weather_client.get_weather_forecast(circuit, days=days_range*2)
                
                if forecast is not None and not forecast.empty:
                    # Save forecast data
                    forecast_path = self._save_data_to_file(
                        forecast, f"{circuit}_forecast",
                        subdir='forecast'
                    )
                    
                    results['forecast'] = forecast_path
                    results['data_paths']['forecast'] = forecast_path
                
                self.agent_logger.task_complete("Fetching weather forecast")
                
                # 3. Get specific race day forecast if race_date is provided
                if race_date:
                    self.agent_logger.task_start("Fetching race day forecast")
                    race_day_forecast = self.weather_client.get_weather_for_race_day(
                        circuit, race_date
                    )
                    
                    if race_day_forecast is not None and not race_day_forecast.empty:
                        # Save race day forecast
                        race_day_path = self._save_data_to_file(
                            race_day_forecast, f"{circuit}_race_day",
                            subdir='forecast'
                        )
                        
                        results['race_day_forecast'] = race_day_path
                        results['data_paths']['race_day_forecast'] = race_day_path
                        
                        # Extract key race conditions for prediction models
                        race_conditions = self._extract_race_conditions(race_day_forecast)
                        results['race_conditions'] = race_conditions
                    
                    self.agent_logger.task_complete("Fetching race day forecast")
                
                # 4. Get historical weather data for this circuit
                self.agent_logger.task_start("Fetching historical weather data")
                
                # Use start and end date from previous years
                historical_data = []
                
                # Get data for the same period in previous 3 years
                for year_offset in range(1, 4):
                    hist_start = start_date.replace(year=start_date.year - year_offset)
                    hist_end = end_date.replace(year=end_date.year - year_offset)
                    
                    # Get historical data for each day in the range
                    current_date = hist_start
                    while current_date <= hist_end:
                        try:
                            hist_weather = self.weather_client.get_historical_weather(
                                circuit, current_date
                            )
                            
                            if hist_weather is not None and not hist_weather.empty:
                                historical_data.append(hist_weather)
                        except Exception as e:
                            self.agent_logger.warning(f"Error fetching historical data for {current_date}: {str(e)}")
                        
                        current_date += timedelta(days=1)
                
                # Combine historical data if available
                if historical_data:
                    combined_historical = pd.concat(historical_data, ignore_index=True)
                    historical_path = self._save_data_to_file(
                        combined_historical, f"{circuit}_historical",
                        subdir='historical'
                    )
                    
                    results['historical_data'] = historical_path
                    results['data_paths']['historical_data'] = historical_path
                
                self.agent_logger.task_complete("Fetching historical weather data")
                
                # 5. Get weather alerts
                self.agent_logger.task_start("Checking weather alerts")
                weather_alerts = self.weather_client.get_weather_alerts(circuit)
                
                if weather_alerts:
                    # Convert to DataFrame for saving
                    alerts_df = pd.DataFrame(weather_alerts)
                    alerts_path = self._save_data_to_file(
                        alerts_df, f"{circuit}_alerts",
                        subdir='alerts'
                    )
                    
                    results['weather_alerts'] = alerts_path
                    results['data_paths']['weather_alerts'] = alerts_path
                    
                    # Also include alerts directly
                    results['active_alerts'] = weather_alerts
                
                self.agent_logger.task_complete("Checking weather alerts")
                
                # 6. Get weather impact probability
                self.agent_logger.task_start("Analyzing weather impact")
                weather_impact = self.weather_client.get_weather_impact_probability(circuit)
                
                if weather_impact:
                    # Convert to DataFrame for saving
                    impact_df = pd.DataFrame([weather_impact])
                    impact_path = self._save_data_to_file(
                        impact_df, f"{circuit}_impact",
                        subdir='forecast'
                    )
                    
                    results['weather_impact'] = impact_path
                    results['data_paths']['weather_impact'] = impact_path
                    
                    # Also include impact data directly
                    results['impact_data'] = weather_impact
                
                self.agent_logger.task_complete("Analyzing weather impact")
                
                # Publish weather monitoring completed event
                self.publish_event("weather_monitoring_completed", {
                    "circuit": circuit,
                    "race_date": results['race_date'],
                    "data_paths": results['data_paths'],
                    "race_conditions": results.get('race_conditions', {}),
                    "active_alerts": results.get('active_alerts', [])
                })
                
                return results
                
            except Exception as e:
                self.agent_logger.error(f"Error during weather monitoring: {str(e)}")
                
                # Publish weather monitoring failed event
                self.publish_event("weather_monitoring_failed", {
                    "circuit": circuit,
                    "race_date": race_date.strftime('%Y-%m-%d') if race_date else None,
                    "error": str(e)
                })
                
                raise
        
    def _save_data_to_file(self, data: pd.DataFrame, name: str, subdir: str = None) -> str:
            """
            Save DataFrame to CSV and JSON files.
            
            Args:
                data: DataFrame to save
                name: Base name for the file
                subdir: Optional subdirectory within data_dir
                
            Returns:
                Path to the saved CSV file
            """
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine directory
            directory = self.data_dir
            if subdir:
                directory = os.path.join(directory, subdir)
                
            # Ensure directory exists
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Create filenames
            base_filename = f"{name}_{timestamp}"
            csv_path = os.path.join(directory, f"{base_filename}.csv")
            json_path = os.path.join(directory, f"{base_filename}.json")
            
            # Save data
            try:
                # Save CSV
                data.to_csv(csv_path, index=False)
                
                # Also save JSON for easier parsing
                # Convert DataFrame to dict for JSON serialization
                data_dict = data.to_dict(orient='records')
                
                with open(json_path, 'w') as f:
                    json.dump(data_dict, f, indent=2, default=str)
                
                self.agent_logger.info(f"Saved weather data to {csv_path} and {json_path}")
                
                return csv_path
                
            except Exception as e:
                self.agent_logger.error(f"Error saving weather data to file: {str(e)}")
                raise
        
    def _extract_race_conditions(self, race_day_forecast: pd.DataFrame) -> Dict[str, Any]:
            """
            Extract key race conditions from the forecast for prediction models.
            
            Args:
                race_day_forecast: DataFrame with race day forecast
                
            Returns:
                Dictionary with key race conditions
            """
            # Initialize with default values
            race_conditions = {
                'weather_is_dry': 1,
                'weather_is_any_wet': 0,
                'weather_is_very_wet': 0,
                'weather_temp_hot': 0,
                'weather_temp_mild': 1,
                'weather_temp_cold': 0,
                'weather_high_wind': 0,
                'temp_celsius': None,
                'wind_speed_ms': None,
                'rain_mm': 0,
                'racing_condition': 'dry'
            }
            
            if race_day_forecast is None or race_day_forecast.empty:
                return race_conditions
            
            try:
                # Filter for race time (typically 14:00-16:00)
                # If race hour is specified, use that instead
                race_hours = [14, 15, 16]  # Typical F1 race hours
                race_time_forecast = race_day_forecast[race_day_forecast['hour'].isin(race_hours)]
                
                if race_time_forecast.empty:
                    # Use all available data if can't filter by race time
                    race_time_forecast = race_day_forecast
                
                # Calculate average conditions during the race
                if 'temp_celsius' in race_time_forecast.columns:
                    race_conditions['temp_celsius'] = race_time_forecast['temp_celsius'].mean()
                    # Temperature categories
                    avg_temp = race_conditions['temp_celsius']
                    if avg_temp < 15:
                        race_conditions['weather_temp_cold'] = 1
                        race_conditions['weather_temp_mild'] = 0
                        race_conditions['weather_temp_hot'] = 0
                    elif avg_temp > 25:
                        race_conditions['weather_temp_cold'] = 0
                        race_conditions['weather_temp_mild'] = 0
                        race_conditions['weather_temp_hot'] = 1
                    else:
                        race_conditions['weather_temp_cold'] = 0
                        race_conditions['weather_temp_mild'] = 1
                        race_conditions['weather_temp_hot'] = 0
                
                if 'wind_speed_ms' in race_time_forecast.columns:
                    race_conditions['wind_speed_ms'] = race_time_forecast['wind_speed_ms'].mean()
                    # High wind flag (above 8 m/s is considered high)
                    race_conditions['weather_high_wind'] = 1 if race_conditions['wind_speed_ms'] >= 8 else 0
                
                if 'rain_1h_mm' in race_time_forecast.columns:
                    race_conditions['rain_mm'] = race_time_forecast['rain_1h_mm'].sum()
                
                # Determine racing condition
                if 'racing_condition' in race_time_forecast.columns:
                    # Get the most common racing condition
                    conditions = race_time_forecast['racing_condition'].value_counts()
                    if not conditions.empty:
                        race_conditions['racing_condition'] = conditions.index[0]
                        
                        # Set weather condition flags
                        if race_conditions['racing_condition'] == 'dry':
                            race_conditions['weather_is_dry'] = 1
                            race_conditions['weather_is_any_wet'] = 0
                            race_conditions['weather_is_very_wet'] = 0
                        elif race_conditions['racing_condition'] in ['damp', 'wet']:
                            race_conditions['weather_is_dry'] = 0
                            race_conditions['weather_is_any_wet'] = 1
                            race_conditions['weather_is_very_wet'] = 0
                        elif race_conditions['racing_condition'] == 'very_wet':
                            race_conditions['weather_is_dry'] = 0
                            race_conditions['weather_is_any_wet'] = 1
                            race_conditions['weather_is_very_wet'] = 1
                
                # Alternative check using precipitation
                elif 'rain_mm' in race_conditions and race_conditions['rain_mm'] is not None:
                    if race_conditions['rain_mm'] > 2.0:
                        race_conditions['weather_is_dry'] = 0
                        race_conditions['weather_is_any_wet'] = 1
                        race_conditions['weather_is_very_wet'] = race_conditions['rain_mm'] > 5.0
                        race_conditions['racing_condition'] = 'very_wet' if race_conditions['rain_mm'] > 5.0 else 'wet'
                
                return race_conditions
                
            except Exception as e:
                self.agent_logger.error(f"Error extracting race conditions: {str(e)}")
                return race_conditions
        
    def monitor_weather_changes(self, circuit: str, race_date: Union[str, datetime], 
                                interval_hours: int = 6) -> None:
            """
            Start continuous monitoring of weather changes for a circuit.
            
            Args:
                circuit: Circuit name/identifier
                race_date: Date of the race
                interval_hours: How often to check for updates
            """
            self.agent_logger.info(f"Starting continuous weather monitoring for {circuit}")
            
            # This would be implemented with a background thread or scheduled task
            # For now, we'll just log the intent
            self.agent_logger.info(
                f"Would monitor weather for {circuit} every {interval_hours} hours until {race_date}"
            )
        
        # In a real implementation, this would set up a scheduler or background task
        # For example, using a scheduler library:
        # 
        # def _scheduled_update():
        #     try:
        #         context = {'circuit': circuit, 'race_date': race_date, 'days_range': 1}
        #         results = self.execute(context)
        #         self.publish_event("weather_update_available", results)
        #     except Exception as e:
        #         self.agent_logger.error(f"Scheduled weather update failed: {str(e)}")
        # 
        # scheduler.add_job(_scheduled_update, 'interval', hours=interval_hours)