"""
Base agent for F1 prediction project.
This module defines the abstract base class that all agents will inherit from.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from crewai import Agent, Task
from agents.utils.communication import MessageBus

# Configure logging
logger = logging.getLogger(__name__)

class F1BaseAgent(ABC):
    """
    Abstract base class for all F1 prediction agents.
    Defines the common interface and functionality that all agents must implement.
    """
    
    def __init__(self, name: str, description: str, goal: str):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            description: Brief description of the agent's role
            goal: The agent's main objective
        """
        self.name = name
        self.description = description
        self.goal = goal
        self.status = "idle"
        self.last_activity = datetime.now()
        self.message_bus = MessageBus()
        
        # Metrics for agent performance monitoring
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0,
            "total_execution_time": 0
        }
        
        logger.info(f"Initialized {self.name} agent with goal: {self.goal}")
    
    def create_crewai_agent(self) -> Agent:
        """
        Create a CrewAI agent from this agent definition.
        
        Returns:
            A CrewAI Agent object
        """
        return Agent(
            name=self.name,
            description=self.description,
            goal=self.goal,
            backstory=self.get_backstory(),
            allow_delegation=True,
            verbose=True
        )
    
    @abstractmethod
    def get_backstory(self) -> str:
        """
        Get the agent's backstory for CrewAI.
        
        Returns:
            String containing the agent's backstory
        """
        pass
        
    @abstractmethod
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the agent's main functionality.
        
        Args:
            context: Optional context with information needed for execution
            
        Returns:
            Dictionary with the results of the execution
        """
        pass
    
    def create_tasks(self) -> List[Task]:
        """
        Create tasks for the CrewAI framework.
        
        Returns:
            List of Task objects for this agent
        """
        # Default implementation that children can override
        return []
    
    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the agent while tracking metrics.
        
        Args:
            context: Optional context with information needed for execution
            
        Returns:
            Dictionary with the results of the execution
        """
        logger.info(f"Starting {self.name} execution")
        self.status = "running"
        self.last_activity = datetime.now()
        
        start_time = time.time()
        
        try:
            # Execute the agent's functionality
            result = self.execute(context)
            
            # Update success metrics
            self.metrics["tasks_completed"] += 1
            execution_time = time.time() - start_time
            self.metrics["total_execution_time"] += execution_time
            self.metrics["avg_execution_time"] = (
                self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
            )
            
            self.status = "completed"
            logger.info(f"{self.name} execution completed in {execution_time:.2f} seconds")
            
            # Publish result to message bus
            self.message_bus.publish(f"{self.name}_completed", {
                "agent": self.name,
                "status": "success",
                "result": result,
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            # Update failure metrics
            self.metrics["tasks_failed"] += 1
            execution_time = time.time() - start_time
            
            self.status = "failed"
            logger.error(f"{self.name} execution failed: {str(e)}")
            
            # Publish failure to message bus
            self.message_bus.publish(f"{self.name}_failed", {
                "agent": self.name,
                "status": "failure",
                "error": str(e),
                "execution_time": execution_time
            })
            
            # Re-raise the exception
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "status": self.status,
            "last_activity": self.last_activity,
            "metrics": self.metrics
        }
    
    def subscribe_to_event(self, event_name: str, callback):
        """
        Subscribe to an event on the message bus.
        
        Args:
            event_name: Name of the event to subscribe to
            callback: Function to call when the event is published
        """
        self.message_bus.subscribe(event_name, callback)
    
    def publish_event(self, event_name: str, data: Dict[str, Any]):
        """
        Publish an event to the message bus.
        
        Args:
            event_name: Name of the event to publish
            data: Data to include with the event
        """
        self.message_bus.publish(event_name, data)