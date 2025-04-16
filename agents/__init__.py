"""
Agent system for F1 prediction project.
This package contains the agent system for automating the F1 prediction workflow.
"""

from agents.base_agent import F1BaseAgent
from agents.supervisor import SupervisorAgent
from agents.data_collector import DataCollectorAgent
from agents.weather_monitor import WeatherMonitorAgent
from agents.prediction_agent import PredictionAgent
from agents.performance_analyzer import PerformanceAnalyzer

__all__ = [
    'F1BaseAgent',
    'SupervisorAgent',
    'DataCollectorAgent',
    'WeatherMonitorAgent',
    'PredictionAgent',
    'PerformanceAnalyzer'
]