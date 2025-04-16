"""
Communication utilities for agents in the F1 prediction project.
Provides classes for agent communication and coordination.
"""

import logging
from typing import Dict, List, Any, Callable
from threading import Lock
import json
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class MessageBus:
    """
    Simple publish-subscribe message bus for inter-agent communication.
    Allows agents to publish events and subscribe to events from other agents.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure all agents use the same message bus instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MessageBus, cls).__new__(cls)
                cls._instance._subscribers = {}
                cls._instance._history = []
                cls._instance._history_size = 100  # Keep last 100 messages
                logger.info("Created new MessageBus instance")
            return cls._instance
    
    def subscribe(self, event_name: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event to subscribe to
            callback: Function to call when the event is published
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        
        self._subscribers[event_name].append(callback)
        logger.debug(f"Subscribed to event: {event_name}")
    
    def publish(self, event_name: str, data: Dict[str, Any]):
        """
        Publish an event to all subscribers.
        
        Args:
            event_name: Name of the event to publish
            data: Data to include with the event
        """
        # Add timestamp and event name to the data
        message = {
            "event": event_name,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Add to history
        self._history.append(message)
        if len(self._history) > self._history_size:
            self._history.pop(0)
        
        # Notify subscribers
        if event_name in self._subscribers:
            for callback in self._subscribers[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback for event {event_name}: {str(e)}")
        
        logger.debug(f"Published event: {event_name}")
    
    def get_history(self, event_name: str = None) -> List[Dict[str, Any]]:
        """
        Get the message history, optionally filtered by event name.
        
        Args:
            event_name: Optional name of event to filter by
            
        Returns:
            List of message dictionaries
        """
        if event_name is None:
            return self._history
        else:
            return [msg for msg in self._history if msg["event"] == event_name]


class NotificationManager:
    """
    Manages notifications for F1 prediction results.
    Can send notifications via various channels (file, email, etc.)
    """
    
    def __init__(self, output_dir: str = "notifications"):
        """
        Initialize the notification manager.
        
        Args:
            output_dir: Directory to store notification files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Subscribe to relevant events
        self.message_bus = MessageBus()
        self.message_bus.subscribe("prediction_completed", self.handle_prediction)
        
        logger.info(f"Initialized NotificationManager with output dir: {output_dir}")
    
    def handle_prediction(self, data: Dict[str, Any]):
        """
        Handle a prediction event and generate appropriate notifications.
        
        Args:
            data: Prediction data
        """
        try:
            # Extract prediction information
            prediction_type = data.get("prediction_type", "unknown")
            race_name = data.get("race_name", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create notification
            notification = {
                "timestamp": datetime.now().isoformat(),
                "prediction_type": prediction_type,
                "race_name": race_name,
                "message": f"New {prediction_type} prediction available for {race_name}",
                "data": data
            }
            
            # Save to file
            filename = f"{timestamp}_{prediction_type}_{race_name.replace(' ', '_')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(notification, f, indent=2)
            
            logger.info(f"Saved notification to {filepath}")
            
        except Exception as e:
            logger.error(f"Error handling prediction notification: {str(e)}")
    
    def send_email_notification(self, to_email: str, subject: str, message: str):
        """
        Send an email notification.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            message: Email message body
        """
        # This is a placeholder for email sending functionality
        # In a real implementation, you would integrate with an email service
        logger.info(f"Would send email to {to_email} with subject '{subject}'")
        
        # Save email to file for demonstration purposes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"{timestamp}_email.txt")
        
        with open(filepath, 'w') as f:
            f.write(f"To: {to_email}\n")
            f.write(f"Subject: {subject}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n\n")
            f.write(message)
        
        logger.info(f"Saved email notification to {filepath}")