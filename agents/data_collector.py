"""
Data collector agent for F1 prediction project.
This agent is responsible for collecting race data via FastF1 API.
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
from api.fastf1_client import FastF1Client

# Configure logging
logger = logging.getLogger(__name__)

class DataCollectorAgent(F1BaseAgent):
    """
    Agent responsible for collecting F1 race data from FastF1 API.
    Gathers race results, qualifying data, practice sessions,
    and historical statistics needed for prediction models.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data collector agent.
        
        Args:
            data_dir: Directory to store collected data
        """
        super().__init__(
            name="Data Collector",
            description="Collects and prepares Formula 1 race data from FastF1 API",
            goal="Gather comprehensive and accurate F1 data for predictive modeling"
        )
        
        self.data_dir = data_dir
        self.fastf1_client = FastF1Client()
        self.agent_logger = AgentLogger(agent_name="DataCollector")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Create subdirectories
        for subdir in ['races', 'qualifying', 'practice', 'historical']:
            path = os.path.join(data_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        self.agent_logger.info(f"Initialized DataCollectorAgent with data directory: {data_dir}")
    
    def get_backstory(self) -> str:
        """
        Get the agent's backstory for CrewAI.
        
        Returns:
            String containing the agent's backstory
        """
        return (
            "I am an expert data engineer specializing in Formula 1 racing data. "
            "I have extensive knowledge of the FastF1 API and know how to extract, "
            "clean, and organize F1 data efficiently. I understand the nuances of "
            "race weekends, from practice sessions to qualifying and race day. "
            "My goal is to provide the most comprehensive and accurate dataset for "
            "building predictive models."
        )
    
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the data collection process.
        
        Args:
            context: Context with information needed for data collection
                Expected keys:
                - year: Season year to collect data for
                - gp_name: Grand Prix name or round number to collect
                - session_types: List of session types to collect ('R', 'Q', 'FP1', 'FP2', 'FP3', 'S')
                - historical_years: Number of historical years to collect
                
        Returns:
            Dictionary with collected data paths
        """
        if context is None:
            context = {}
        
        # Extract parameters from context
        year = context.get('year', datetime.now().year)
        gp_name = context.get('gp_name')
        session_types = context.get('session_types', ['R', 'Q', 'FP1', 'FP2', 'FP3'])
        historical_years = context.get('historical_years', 3)
        
        self.agent_logger.info(f"Starting data collection for: Year={year}, GP={gp_name}")
        
        # Validate input
        if gp_name is None:
            # Try to determine the next race if not specified
            try:
                calendar = self.fastf1_client.get_race_calendar(year)
                for _, race in calendar.iterrows():
                    race_date = pd.to_datetime(race['EventDate'])
                    if race_date > datetime.now():
                        gp_name = race['EventName']
                        self.agent_logger.info(f"Automatically selected next race: {gp_name}")
                        break
                
                if gp_name is None:
                    self.agent_logger.error("Could not determine next race")
                    raise ValueError("No Grand Prix specified and could not determine next race")
            except Exception as e:
                self.agent_logger.error(f"Error determining next race: {str(e)}")
                raise
        
        # Store results
        results = {
            'year': year,
            'gp_name': gp_name,
            'data_paths': {},
            'calendar': None,
            'sessions': {},
            'historical_data': {},
            'driver_standings': None,
            'constructor_standings': None
        }
        
        try:
            # 1. Fetch race calendar
            self.agent_logger.task_start("Fetching race calendar")
            calendar = self.fastf1_client.get_race_calendar(year)
            calendar_path = self._save_data_to_file(calendar, 'calendar', year=year)
            results['calendar'] = calendar_path
            results['data_paths']['calendar'] = calendar_path
            self.agent_logger.task_complete("Fetching race calendar")
            
            # Find the specified Grand Prix in the calendar
            gp_info = None
            for _, race in calendar.iterrows():
                if race['EventName'] == gp_name:
                    gp_info = race
                    break
            
            # 2. Fetch current standings
            self.agent_logger.task_start("Fetching current standings")
            driver_standings = self.fastf1_client.get_driver_standings(year)
            driver_standings_path = self._save_data_to_file(
                driver_standings, 'driver_standings', year=year
            )
            
            constructor_standings = self.fastf1_client.get_constructor_standings(year)
            constructor_standings_path = self._save_data_to_file(
                constructor_standings, 'constructor_standings', year=year
            )
            
            results['driver_standings'] = driver_standings_path
            results['constructor_standings'] = constructor_standings_path
            results['data_paths']['driver_standings'] = driver_standings_path
            results['data_paths']['constructor_standings'] = constructor_standings_path
            self.agent_logger.task_complete("Fetching current standings")
            
            # 3. Fetch specific Grand Prix data
            for session_type in session_types:
                self.agent_logger.task_start(f"Fetching {session_type} data for {gp_name}")
                
                # Check if the session exists
                try:
                    session_data = self.fastf1_client.get_session(year, gp_name, session_type)
                    
                    # Get session results
                    if session_type == 'R':
                        data = self.fastf1_client.get_race_results(year, gp_name)
                        subdir = 'races'
                    elif session_type == 'Q':
                        data = self.fastf1_client.get_qualifying_results(year, gp_name)
                        subdir = 'qualifying'
                    elif session_type == 'S':
                        data = self.fastf1_client.get_sprint_results(year, gp_name)
                        subdir = 'races'
                    else:  # Practice sessions
                        data = self.fastf1_client.get_session_lap_data(year, gp_name, session_type)
                        subdir = 'practice'
                    
                    if data is not None and not data.empty:
                        # Save data to file
                        file_path = self._save_data_to_file(
                            data, f"{gp_name.replace(' ', '_')}_{session_type}",
                            year=year, subdir=subdir
                        )
                        
                        results['sessions'][session_type] = file_path
                        results['data_paths'][f"{gp_name}_{session_type}"] = file_path
                    else:
                        self.agent_logger.warning(f"No data available for {session_type} session")
                        
                except Exception as e:
                    self.agent_logger.error(f"Error fetching {session_type} data: {str(e)}")
                    # Continue with the next session type
                
                self.agent_logger.task_complete(f"Fetching {session_type} data for {gp_name}")
            
            # 4. Fetch historical data for this circuit
            if gp_info is not None:
                circuit_name = gp_info['EventName']
                self.agent_logger.task_start(f"Fetching historical data for {circuit_name}")
                
                historical_data = []
                
                # Collect data from previous years
                for hist_year in range(year - historical_years, year):
                    try:
                        # Check if there was a race at this circuit in the historical year
                        hist_calendar = self.fastf1_client.get_race_calendar(hist_year)
                        circuit_found = False
                        
                        for _, hist_race in hist_calendar.iterrows():
                            # Try to match by circuit name
                            if hist_race['EventName'] == circuit_name:
                                circuit_found = True
                                hist_gp = hist_race['EventName']
                                break
                        
                        if not circuit_found:
                            self.agent_logger.warning(f"No race found at {circuit_name} in {hist_year}")
                            continue
                        
                        # Get historical race results
                        hist_results = self.fastf1_client.get_race_results(hist_year, hist_gp)
                        if hist_results is not None and not hist_results.empty:
                            historical_data.append(hist_results)
                            
                            # Save individual historical data
                            file_path = self._save_data_to_file(
                                hist_results, f"{hist_gp.replace(' ', '_')}_R",
                                year=hist_year, subdir='historical'
                            )
                            
                            results['historical_data'][f"{hist_year}_{hist_gp}"] = file_path
                            results['data_paths'][f"historical_{hist_year}_{hist_gp}"] = file_path
                        
                    except Exception as e:
                        self.agent_logger.error(f"Error fetching historical data for {hist_year}: {str(e)}")
                
                # Combine historical data if available
                if historical_data:
                    combined_historical = pd.concat(historical_data, ignore_index=True)
                    file_path = self._save_data_to_file(
                        combined_historical, f"{circuit_name.replace(' ', '_')}_historical",
                        year=year, subdir='historical'
                    )
                    
                    results['historical_data']['combined'] = file_path
                    results['data_paths'][f"historical_combined_{circuit_name}"] = file_path
                
                self.agent_logger.task_complete(f"Fetching historical data for {circuit_name}")
            
            # Publish data collection completed event
            self.publish_event("data_collection_completed", {
                "year": year,
                "gp_name": gp_name,
                "data_paths": results['data_paths']
            })
            
            return results
            
        except Exception as e:
            self.agent_logger.error(f"Error during data collection: {str(e)}")
            
            # Publish data collection failed event
            self.publish_event("data_collection_failed", {
                "year": year,
                "gp_name": gp_name,
                "error": str(e)
            })
            
            raise
    
    def _save_data_to_file(self, data: pd.DataFrame, name: str, year: int, 
                          subdir: str = None) -> str:
        """
        Save DataFrame to CSV and JSON files.
        
        Args:
            data: DataFrame to save
            name: Base name for the file
            year: Year of the data
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
        base_filename = f"{year}_{name}_{timestamp}"
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
            
            self.agent_logger.info(f"Saved data to {csv_path} and {json_path}")
            
            return csv_path
            
        except Exception as e:
            self.agent_logger.error(f"Error saving data to file: {str(e)}")
            raise
    
    def collect_driver_data(self, year: int, drivers: List[str] = None) -> Dict[str, Any]:
        """
        Collect data for specific drivers.
        
        Args:
            year: Season year to collect data for
            drivers: List of driver codes to collect data for (None for all drivers)
            
        Returns:
            Dictionary with collected driver data
        """
        self.agent_logger.task_start(f"Collecting driver data for {year}")
        
        try:
            # Get driver info
            driver_info = self.fastf1_client.get_driver_info(year)
            
            # If no specific drivers requested, use all drivers
            if drivers is None and driver_info is not None:
                drivers = driver_info['Abbreviation'].tolist()
            
            results = {
                'year': year,
                'drivers': drivers,
                'data_paths': {},
                'driver_info': {}
            }
            
            # Get driver standings
            driver_standings = self.fastf1_client.get_driver_standings(year)
            if driver_standings is not None:
                standings_path = self._save_data_to_file(driver_standings, 'driver_standings', year=year)
                results['data_paths']['driver_standings'] = standings_path
            
            # Collect data for each driver
            for driver in drivers:
                self.agent_logger.info(f"Collecting data for driver: {driver}")
                
                driver_results = {}
                
                # Get race calendar
                calendar = self.fastf1_client.get_race_calendar(year)
                
                # Collect results for each race
                for _, race in calendar.iterrows():
                    gp_name = race['EventName']
                    race_date = pd.to_datetime(race['EventDate'])
                    
                    # Skip future races
                    if race_date > datetime.now():
                        continue
                    
                    try:
                        # Get race results
                        race_results = self.fastf1_client.get_race_results(year, gp_name)
                        if race_results is not None and not race_results.empty:
                            # Filter for this driver
                            driver_race = race_results[race_results['Driver'] == driver]
                            if not driver_race.empty:
                                driver_results[gp_name] = {
                                    'race': driver_race.to_dict('records')[0]
                                }
                        
                        # Get qualifying results
                        quali_results = self.fastf1_client.get_qualifying_results(year, gp_name)
                        if quali_results is not None and not quali_results.empty:
                            # Filter for this driver
                            driver_quali = quali_results[quali_results['Driver'] == driver]
                            if not driver_quali.empty and gp_name in driver_results:
                                driver_results[gp_name]['qualifying'] = driver_quali.to_dict('records')[0]
                        
                    except Exception as e:
                        self.agent_logger.warning(f"Error collecting {gp_name} data for {driver}: {str(e)}")
                
                # Save driver season data
                driver_season_df = pd.DataFrame([
                    {
                        'GP': gp,
                        'QualifyingPosition': data.get('qualifying', {}).get('Position', None),
                        'GridPosition': data.get('race', {}).get('GridPosition', None),
                        'RacePosition': data.get('race', {}).get('Position', None),
                        'Points': data.get('race', {}).get('Points', 0),
                        'Status': data.get('race', {}).get('Status', None)
                    }
                    for gp, data in driver_results.items()
                ])
                
                if not driver_season_df.empty:
                    file_path = self._save_data_to_file(
                        driver_season_df, f"driver_{driver}_season",
                        year=year, subdir='drivers'
                    )
                    
                    results['data_paths'][f"driver_{driver}"] = file_path
                    results['driver_info'][driver] = {
                        'season_data_path': file_path,
                        'races': list(driver_results.keys())
                    }
            
            self.agent_logger.task_complete(f"Collecting driver data for {year}")
            return results
            
        except Exception as e:
            self.agent_logger.task_fail(f"Collecting driver data for {year}", str(e))
            raise
    
    def collect_team_data(self, year: int, teams: List[str] = None) -> Dict[str, Any]:
        """
        Collect data for specific teams.
        
        Args:
            year: Season year to collect data for
            teams: List of team names to collect data for (None for all teams)
            
        Returns:
            Dictionary with collected team data
        """
        self.agent_logger.task_start(f"Collecting team data for {year}")
        
        try:
            # Get constructor standings
            constructor_standings = self.fastf1_client.get_constructor_standings(year)
            
            # If no specific teams requested, use all teams
            if teams is None and constructor_standings is not None:
                teams = constructor_standings['Team'].unique().tolist()
            
            results = {
                'year': year,
                'teams': teams,
                'data_paths': {},
                'team_info': {}
            }
            
            # Save constructor standings
            if constructor_standings is not None:
                standings_path = self._save_data_to_file(
                    constructor_standings, 'constructor_standings',
                    year=year
                )
                results['data_paths']['constructor_standings'] = standings_path
            
            # Get race calendar
            calendar = self.fastf1_client.get_race_calendar(year)
            
            # Collect data for each team
            for team in teams:
                self.agent_logger.info(f"Collecting data for team: {team}")
                
                team_results = {}
                
                # Collect results for each race
                for _, race in calendar.iterrows():
                    gp_name = race['EventName']
                    race_date = pd.to_datetime(race['EventDate'])
                    
                    # Skip future races
                    if race_date > datetime.now():
                        continue
                    
                    try:
                        # Get race results
                        race_results = self.fastf1_client.get_race_results(year, gp_name)
                        if race_results is not None and not race_results.empty:
                            # Filter for this team
                            team_race = race_results[race_results['Team'] == team]
                            if not team_race.empty:
                                team_results[gp_name] = {
                                    'race': team_race.to_dict('records')
                                }
                        
                        # Get qualifying results
                        quali_results = self.fastf1_client.get_qualifying_results(year, gp_name)
                        if quali_results is not None and not quali_results.empty:
                            # Filter for this team
                            team_quali = quali_results[quali_results['Team'] == team]
                            if not team_quali.empty and gp_name in team_results:
                                team_results[gp_name]['qualifying'] = team_quali.to_dict('records')
                        
                    except Exception as e:
                        self.agent_logger.warning(f"Error collecting {gp_name} data for {team}: {str(e)}")
                
                # Process and save team season data
                if team_results:
                    # Calculate team performance metrics for each race
                    team_performance = []
                    
                    for gp, data in team_results.items():
                        race_data = data.get('race', [])
                        quali_data = data.get('qualifying', [])
                        
                        if race_data:
                            # Calculate average position, points, etc.
                            avg_position = sum(d.get('Position', 0) for d in race_data) / len(race_data)
                            total_points = sum(d.get('Points', 0) for d in race_data)
                            best_position = min((d.get('Position', float('inf')) for d in race_data), default=None)
                            
                            # Calculate average qualifying position
                            avg_quali_position = None
                            if quali_data:
                                avg_quali_position = sum(d.get('Position', 0) for d in quali_data) / len(quali_data)
                            
                            team_performance.append({
                                'GP': gp,
                                'AvgPosition': avg_position,
                                'BestPosition': best_position,
                                'TotalPoints': total_points,
                                'AvgQualifyingPosition': avg_quali_position,
                                'DriversFinished': sum(1 for d in race_data if 'Finished' in str(d.get('Status', '')))
                            })
                    
                    # Create and save DataFrame
                    if team_performance:
                        team_df = pd.DataFrame(team_performance)
                        file_path = self._save_data_to_file(
                            team_df, f"team_{team.replace(' ', '_')}_season",
                            year=year, subdir='teams'
                        )
                        
                        results['data_paths'][f"team_{team.replace(' ', '_')}"] = file_path
                        results['team_info'][team] = {
                            'season_data_path': file_path,
                            'races': list(team_results.keys())
                        }
            
            self.agent_logger.task_complete(f"Collecting team data for {year}")
            return results
            
        except Exception as e:
            self.agent_logger.task_fail(f"Collecting team data for {year}", str(e))
            raise