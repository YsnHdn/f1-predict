"""
Supervisor agent for F1 prediction project.
This agent coordinates the workflow of all other agents and manages the prediction process.
"""

import logging
import os
import json
import pandas as pd
import fastf1
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from crewai import Crew, Process, Task
from crewai.agent import Agent

from agents.base_agent import F1BaseAgent
from agents.utils.logging import AgentLogger
from agents.utils.communication import MessageBus, NotificationManager
from agents.data_collector import DataCollectorAgent
from agents.weather_monitor import WeatherMonitorAgent
from agents.prediction_agent import PredictionAgent
from agents.performance_analyzer import PerformanceAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class SupervisorAgent(F1BaseAgent):
    """
    Agent responsible for coordinating the work of all other agents.
    Manages the prediction process, schedules tasks, and ensures all components work together.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the supervisor agent.
        
        Args:
            output_dir: Directory to store outputs
        """
        super().__init__(
            name="Supervisor",
            description="Coordinates the F1 prediction process across all agents",
            goal="Ensure seamless collaboration between agents for timely and accurate race predictions"
        )
        
        self.output_dir = output_dir
        self.agent_logger = AgentLogger(agent_name="Supervisor")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Initialize notification manager
        self.notification_manager = NotificationManager()
        
        # Initialize other agents
        self.data_collector = DataCollectorAgent()
        self.weather_monitor = WeatherMonitorAgent()
        self.prediction_agent = PredictionAgent()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Subscribe to agent events
        self._setup_event_subscriptions()
        
        # State tracking
        self.current_race = None
        self.prediction_status = {
            'initial': False,
            'pre_race': False,
            'race_day': False
        }
        self.data_paths = {}
        
        self.agent_logger.info("Initialized SupervisorAgent")
    
    def get_backstory(self) -> str:
        """
        Get the agent's backstory for CrewAI.
        
        Returns:
            String containing the agent's backstory
        """
        return (
            "I am the chief strategist for a Formula 1 prediction system. With decades of experience "
            "in race management and data operations, I ensure all components of our prediction system "
            "work together seamlessly. I understand the timing and coordination required for successful "
            "predictions throughout a race weekend. My expertise lies in orchestrating the collection "
            "of data, monitoring of conditions, generation of predictions, and delivery of insights "
            "to users at precisely the right moments."
        )
    
    def _get_next_race(self, year: int) -> Dict[str, Any]:
        """
        Récupère automatiquement la prochaine course pour l'année spécifiée.
        
        Args:
            year: Année pour laquelle trouver la prochaine course
        
        Returns:
            Dictionnaire avec les informations de la prochaine course
        """
        try:
            # Récupérer le calendrier des courses pour l'année
            schedule = fastf1.get_event_schedule(year)
            
            # Filtrer les courses futures à partir de maintenant
            future_races = schedule[schedule['EventDate'] > datetime.now()]
            
            if future_races.empty:
                # Si aucune course future n'est trouvée cette année, lever une exception
                raise ValueError(f"Aucune course future trouvée pour l'année {year}")
            
            # Prendre la première course future
            next_race = future_races.iloc[0]
            
            return {
                'race_name': next_race['EventName'],
                'circuit': next_race['OfficialEventName'],  # Utilisez 'OfficialEventName' au lieu de 'Circuit'
                'race_date': next_race['EventDate']
            }
        
        except Exception as e:
            self.agent_logger.error(f"Erreur lors de la récupération de la prochaine course : {e}")
            raise
        
    def _setup_event_subscriptions(self):
        """Set up subscriptions to events from other agents."""
        # Subscribe to data collection events
        self.subscribe_to_event("data_collection_completed", self._handle_data_collection_completed)
        self.subscribe_to_event("data_collection_failed", self._handle_data_collection_failed)
        
        # Subscribe to weather monitoring events
        self.subscribe_to_event("weather_monitoring_completed", self._handle_weather_monitoring_completed)
        self.subscribe_to_event("weather_monitoring_failed", self._handle_weather_monitoring_failed)
        
        # Subscribe to prediction events
        self.subscribe_to_event("prediction_completed", self._handle_prediction_completed)
        self.subscribe_to_event("prediction_failed", self._handle_prediction_failed)
        
        # Subscribe to analysis events
        self.subscribe_to_event("analysis_completed", self._handle_analysis_completed)
        self.subscribe_to_event("analysis_failed", self._handle_analysis_failed)
    
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the main supervisor workflow.
        
        Args:
            context: Context with information needed for coordinating agents
                Expected keys:
                - race_name (optional): Name of the race
                - circuit (optional): Circuit identifier
                - race_date (optional): Date of the race
                - year (optional): Year to find next race
                - prediction_types: List of prediction types to run ('initial', 'pre_race', 'race_day')
                
        Returns:
            Dictionary with workflow results
        """
        if context is None:
            context = {}
        
        # Récupérer l'année depuis le contexte, par défaut l'année courante
        year = context.get('year', datetime.now().year)
        prediction_types = context.get('prediction_types', ['initial', 'pre_race', 'race_day'])
        
        # Si race_name n'est pas spécifié, le récupérer automatiquement
        if not context.get('race_name'):
            try:
                next_race_info = self._get_next_race(year)
                context.update(next_race_info)
            except ValueError as e:
                self.agent_logger.error(str(e))
                raise
        
        # Extract parameters from context
        race_name = context['race_name']
        circuit = context.get('circuit')
        race_date = context.get('race_date')
        
        # Validate inputs
        if not race_name:
            self.agent_logger.error("No race name specified")
            raise ValueError("Race name is required")
        
        if not circuit:
            self.agent_logger.error("No circuit specified")
            raise ValueError("Circuit identifier is required")
        
        if not race_date:
            self.agent_logger.error("No race date specified")
            raise ValueError("Race date is required")
        
        # Convert race_date string to datetime if needed
        if isinstance(race_date, str):
            try:
                race_date = datetime.strptime(race_date, '%Y-%m-%d')
            except ValueError:
                self.agent_logger.error(f"Invalid race date format: {race_date}")
                raise ValueError("Race date must be in YYYY-MM-DD format")
        
        # Update current race information
        self.current_race = {
            'name': race_name,
            'circuit': circuit,
            'date': race_date.strftime('%Y-%m-%d'),
            'year': race_date.year
        }
        
        # Reset prediction status
        self.prediction_status = {
            'initial': False,
            'pre_race': False,
            'race_day': False
        }
        
        self.agent_logger.info(f"Starting prediction workflow for {race_name} at {circuit} on {race_date.strftime('%Y-%m-%d')}")
        
        # Store results
        results = {
            'race_info': self.current_race,
            'workflow_status': 'in_progress',
            'data_collection': None,
            'weather_monitoring': None,
            'predictions': {},
            'analyses': {},
            'errors': []
        }
        
        try:
            # Step 1: Collect data
            self.agent_logger.task_start("Collecting race data")
            data_collection_context = {
                'year': race_date.year,
                'gp_name': race_name,
                'session_types': ['R', 'Q', 'FP1', 'FP2', 'FP3'],
                'historical_years': 3
            }
            
            data_collection_results = self.data_collector.run(data_collection_context)
            results['data_collection'] = data_collection_results
            self.data_paths = data_collection_results.get('data_paths', {})
            self.agent_logger.task_complete("Collecting race data")
            
            # Step 2: Monitor weather
            self.agent_logger.task_start("Monitoring weather conditions")
            weather_context = {
                'circuit': circuit,
                'race_date': race_date,
                'days_range': 3
            }
            
            weather_results = self.weather_monitor.run(weather_context)
            results['weather_monitoring'] = weather_results
            self.agent_logger.task_complete("Monitoring weather conditions")
            
            # Step 3: Generate predictions based on requested types
            weather_conditions = weather_results.get('race_conditions', {})
            
            for prediction_type in prediction_types:
                self.agent_logger.task_start(f"Generating {prediction_type} prediction")
                
                prediction_context = {
                    'prediction_type': prediction_type,
                    'data_paths': self.data_paths,
                    'race_info': self.current_race,
                    'weather_conditions': weather_conditions
                }
                
                try:
                    prediction_results = self.prediction_agent.run(prediction_context)
                    results['predictions'][prediction_type] = prediction_results
                    self.prediction_status[prediction_type] = True
                    self.agent_logger.task_complete(f"Generating {prediction_type} prediction")
                except Exception as e:
                    self.agent_logger.task_fail(f"Generating {prediction_type} prediction", str(e))
                    results['errors'].append({
                        'step': f"{prediction_type}_prediction",
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Step 4: Analyze prediction performance (for historical races only)
            if race_date < datetime.now():
                self.agent_logger.task_start("Analyzing prediction performance")
                
                analysis_context = {
                    'race_info': self.current_race,
                    'data_paths': self.data_paths,
                    'predictions': results['predictions']
                }
                
                try:
                    analysis_results = self.performance_analyzer.run(analysis_context)
                    results['analyses'] = analysis_results
                    self.agent_logger.task_complete("Analyzing prediction performance")
                except Exception as e:
                    self.agent_logger.task_fail("Analyzing prediction performance", str(e))
                    results['errors'].append({
                        'step': "performance_analysis",
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Save workflow results
            results['workflow_status'] = 'completed'
            self._save_workflow_results(results)
            
            # Publish workflow completed event
            self.publish_event("workflow_completed", {
                "race_name": race_name,
                "prediction_types": prediction_types,
                "prediction_status": self.prediction_status
            })
            
            return results
            
        except Exception as e:
            self.agent_logger.error(f"Error in prediction workflow: {str(e)}")
            
            # Update status
            results['workflow_status'] = 'failed'
            results['errors'].append({
                'step': "workflow",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            # Save partial results
            self._save_workflow_results(results)
            
            # Publish workflow failed event
            self.publish_event("workflow_failed", {
                "race_name": race_name,
                "error": str(e)
            })
            
            raise
    
    def _save_workflow_results(self, results: Dict[str, Any]) -> str:
        """
        Save workflow results to file.
        
        Args:
            results: Workflow results to save
            
        Returns:
            Path to the saved file
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get race name
        race_name = results['race_info'].get('name', 'unknown').replace(' ', '_')
        
        # Create filename
        filename = f"{timestamp}_workflow_{race_name}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save data
        try:
            # Filter out non-serializable objects
            def sanitize(obj):
                if isinstance(obj, pd.DataFrame):
                    return "DataFrame(rows={}, cols={})".format(len(obj), len(obj.columns))
                return str(obj)
            
            # Save JSON
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=sanitize)
            
            self.agent_logger.info(f"Saved workflow results to {filepath}")
            
            return filepath
            
        except Exception as e:
            self.agent_logger.error(f"Error saving workflow results: {str(e)}")
            return None
    
    def _handle_data_collection_completed(self, data: Dict[str, Any]):
        """Handle data collection completed event."""
        self.agent_logger.info(f"Data collection completed for {data.get('gp_name', 'unknown race')}")
        self.data_paths.update(data.get('data_paths', {}))
    
    def _handle_data_collection_failed(self, data: Dict[str, Any]):
        """Handle data collection failed event."""
        self.agent_logger.error(f"Data collection failed for {data.get('gp_name', 'unknown race')}: {data.get('error', 'unknown error')}")
    
    def _handle_weather_monitoring_completed(self, data: Dict[str, Any]):
        """Handle weather monitoring completed event."""
        self.agent_logger.info(f"Weather monitoring completed for {data.get('circuit', 'unknown circuit')}")
        
        # If there are any active alerts, log them
        active_alerts = data.get('active_alerts', [])
        if active_alerts:
            alerts_msg = ", ".join([alert.get('event', 'unknown') for alert in active_alerts])
            self.agent_logger.warning(f"Weather alerts active: {alerts_msg}")
    
    def _handle_weather_monitoring_failed(self, data: Dict[str, Any]):
        """Handle weather monitoring failed event."""
        self.agent_logger.error(f"Weather monitoring failed for {data.get('circuit', 'unknown circuit')}: {data.get('error', 'unknown error')}")
    
    def _handle_prediction_completed(self, data: Dict[str, Any]):
        """Handle prediction completed event."""
        prediction_type = data.get('prediction_type', 'unknown')
        race_name = data.get('race_name', 'unknown race')
        self.agent_logger.info(f"{prediction_type} prediction completed for {race_name}")
        
        # Update prediction status
        if prediction_type in self.prediction_status:
            self.prediction_status[prediction_type] = True
        
        # Get summary for logging
        summary = data.get('predictions_summary', {})
        if 'podium' in summary:
            podium = ", ".join(summary['podium'])
            self.agent_logger.info(f"Predicted podium: {podium}")
    
    def _handle_prediction_failed(self, data: Dict[str, Any]):
        """Handle prediction failed event."""
        prediction_type = data.get('prediction_type', 'unknown')
        race_name = data.get('race_name', 'unknown race')
        error = data.get('error', 'unknown error')
        self.agent_logger.error(f"{prediction_type} prediction failed for {race_name}: {error}")
    
    def _handle_analysis_completed(self, data: Dict[str, Any]):
        """Handle analysis completed event."""
        self.agent_logger.info(f"Performance analysis completed for {data.get('race_name', 'unknown race')}")
    
    def _handle_analysis_failed(self, data: Dict[str, Any]):
        """Handle analysis failed event."""
        self.agent_logger.error(f"Performance analysis failed for {data.get('race_name', 'unknown race')}: {data.get('error', 'unknown error')}")
    
    def create_crewai_crew(self) -> Crew:
        """
        Create a CrewAI crew with all agents.
        
        Returns:
            CrewAI Crew object
        """
        # Create CrewAI agents
        data_collector_agent = self.data_collector.create_crewai_agent()
        weather_agent = self.weather_monitor.create_crewai_agent()
        prediction_agent = self.prediction_agent.create_crewai_agent()
        
        # Create tasks for each agent
        data_collection_task = Task(
            description="Collect all necessary data for the current race from the FastF1 API",
            agent=data_collector_agent,
            expected_output="Complete dataset including race results, qualifying data, and historical statistics"
        )
        
        weather_task = Task(
            description="Monitor and analyze weather conditions for the race circuit",
            agent=weather_agent,
            expected_output="Weather forecast, historical weather data, and race day condition predictions"
        )
        
        initial_prediction_task = Task(
            description="Generate initial race predictions based on historical data",
            agent=prediction_agent,
            expected_output="Initial driver position predictions before qualifying"
        )
        
        pre_race_prediction_task = Task(
            description="Update predictions after qualifying with grid position information",
            agent=prediction_agent,
            expected_output="Pre-race driver position predictions after qualifying"
        )
        
        race_day_prediction_task = Task(
            description="Generate final race predictions with up-to-date weather and last-minute information",
            agent=prediction_agent,
            expected_output="Final driver position predictions for the race"
        )
        
        # Create the crew
        crew = Crew(
            agents=[data_collector_agent, weather_agent, prediction_agent],
            tasks=[data_collection_task, weather_task, initial_prediction_task, 
                  pre_race_prediction_task, race_day_prediction_task],
            verbose=True,
            process=Process.sequential  # Tasks will run in sequence
        )
        
        return crew
    
    def run_crewai_workflow(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the prediction workflow using CrewAI.
        
        Args:
            context: Context with information needed for coordinating agents
                
        Returns:
            Dictionary with workflow results
        """
        if context is None:
            context = {}
        
        # Extract parameters from context
        race_name = context.get('race_name')
        circuit = context.get('circuit')
        race_date = context.get('race_date')
        
        # Validate inputs
        if not race_name or not circuit or not race_date:
            self.agent_logger.error("Missing required parameters")
            raise ValueError("race_name, circuit, and race_date are required")
        
        self.agent_logger.info(f"Starting CrewAI workflow for {race_name}")
        
        # Create and run the crew
        crew = self.create_crewai_crew()
        result = crew.kickoff()
        
        self.agent_logger.info(f"CrewAI workflow completed for {race_name}")
        
        return {
            'race_info': {
                'name': race_name,
                'circuit': circuit,
                'date': race_date
            },
            'crewai_result': result
        }
        
    def schedule_predictions(self, race_schedule: Dict[str, Any]) -> None:
        """
        Schedule predictions for upcoming races.
        
        Args:
            race_schedule: Dictionary with race schedule information
                Expected format:
                {
                    'races': [
                        {
                            'name': 'Race Name',
                            'circuit': 'circuit_id',
                            'date': '2023-04-15',
                            'qualifying_date': '2023-04-14',
                            'fp1_date': '2023-04-13',
                            'fp2_date': '2023-04-13',
                            'fp3_date': '2023-04-14'
                        },
                        ...
                    ]
                }
        """
        self.agent_logger.info("Scheduling predictions for upcoming races")
        
        # In a real implementation, this would set up scheduled tasks
        # For example, using a scheduler library:
        # 
        # for race in race_schedule.get('races', []):
        #     race_name = race.get('name')
        #     circuit = race.get('circuit')
        #     race_date = datetime.strptime(race.get('date'), '%Y-%m-%d')
        #     qualifying_date = datetime.strptime(race.get('qualifying_date'), '%Y-%m-%d')
        #     fp1_date = datetime.strptime(race.get('fp1_date'), '%Y-%m-%d')
        # 
        #     # Schedule initial prediction after FP1
        #     initial_prediction_time = fp1_date + timedelta(hours=2)
        #     scheduler.add_job(
        #         self.execute,
        #         'date',
        #         run_date=initial_prediction_time,
        #         kwargs={
        #             'context': {
        #                 'race_name': race_name,
        #                 'circuit': circuit,
        #                 'race_date': race_date.strftime('%Y-%m-%d'),
        #                 'prediction_types': ['initial']
        #             }
        #         }
        #     )
        # 
        #     # Schedule pre-race prediction after qualifying
        #     pre_race_prediction_time = qualifying_date + timedelta(hours=2)
        #     scheduler.add_job(...)
        # 
        #     # Schedule race day prediction 3 hours before race
        #     race_day_prediction_time = race_date - timedelta(hours=3)
        #     scheduler.add_job(...)
        
        # For now, just log the intent
        for race in race_schedule.get('races', []):
            race_name = race.get('name')
            circuit = race.get('circuit')
            race_date = race.get('date')
            qualifying_date = race.get('qualifying_date')
            
            self.agent_logger.info(f"Would schedule predictions for {race_name} at {circuit} on {race_date}")
            self.agent_logger.info(f"  - Initial prediction after FP1")
            self.agent_logger.info(f"  - Pre-race prediction after qualifying on {qualifying_date}")
            self.agent_logger.info(f"  - Race day prediction 3 hours before race on {race_date}")

# Fin du fichier supervisor.py