"""
Logging utilities for agents in the F1 prediction project.
Provides consistent logging setup and formats across agents.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Name of the logger
        log_file: Optional file path for log output
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class AgentLogger:
    """
    Logger class specifically for agents, with additional agent-specific context.
    """
    
    def __init__(self, agent_name: str, log_dir: str = "logs"):
        """
        Initialize the agent logger.
        
        Args:
            agent_name: Name of the agent
            log_dir: Directory for log files
        """
        self.agent_name = agent_name
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{timestamp}_{agent_name}.log")
        
        # Set up logger
        self.logger = setup_logger(f"agent.{agent_name}", log_file)
        
        self.logger.info(f"Initialized logger for agent: {agent_name}")
    
    def info(self, message: str):
        """Log an info message."""
        self.logger.info(f"[{self.agent_name}] {message}")
    
    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(f"[{self.agent_name}] {message}")
    
    def error(self, message: str):
        """Log an error message."""
        self.logger.error(f"[{self.agent_name}] {message}")
    
    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(f"[{self.agent_name}] {message}")
    
    def critical(self, message: str):
        """Log a critical message."""
        self.logger.critical(f"[{self.agent_name}] {message}")
    
    def task_start(self, task_name: str):
        """Log the start of a task."""
        self.logger.info(f"[{self.agent_name}] Starting task: {task_name}")
    
    def task_complete(self, task_name: str):
        """Log the completion of a task."""
        self.logger.info(f"[{self.agent_name}] Completed task: {task_name}")
    
    def task_fail(self, task_name: str, error: str):
        """Log the failure of a task."""
        self.logger.error(f"[{self.agent_name}] Failed task: {task_name} - {error}")