"""
Base model module for F1 prediction project.
This module defines the abstract base class that all prediction models will inherit from.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logger = logging.getLogger(__name__)

class F1PredictionModel(ABC, BaseEstimator):
    """
    Abstract base class for all F1 prediction models.
    Defines the common interface and functionality that all models must implement.
    """
    
    def __init__(self, name: str = 'base_model', version: str = '0.1'):
        """
        Initialize the base prediction model.
        
        Args:
            name: Name of the model
            version: Version string
        """
        self.name = name
        self.version = version
        self.model = None
        self.features = []
        self.target = ''
        self.is_trained = False
        self.training_date = None
        self.metrics = {}
        
        logger.info(f"Initialized {self.name} model (version {self.version})")
    
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the model from the input DataFrame.
        This method should be implemented by subclasses to select and transform
        the features needed for this specific model.
        
        Args:
            df: Input DataFrame with all available data
            
        Returns:
            DataFrame with selected and prepared features
        """
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model using the provided features and target.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance on the provided data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error("Model is not trained, cannot evaluate")
            return {}
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
            
            # Store metrics
            self.metrics = metrics
            
            logger.info(f"Model evaluation: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return {}
    
    def save(self, filepath: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import joblib
            
            # Create a dictionary with model state
            model_data = {
                'name': self.name,
                'version': self.version,
                'model': self.model,
                'features': self.features,
                'target': self.target,
                'is_trained': self.is_trained,
                'training_date': self.training_date,
                'metrics': self.metrics
            }
            
            # Save to disk
            joblib.dump(model_data, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, filepath: str) -> bool:
        """
        Load the model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import joblib
            
            # Load from disk
            model_data = joblib.load(filepath)
            
            # Update model state
            self.name = model_data['name']
            self.version = model_data['version']
            self.model = model_data['model']
            self.features = model_data['features']
            self.target = model_data['target']
            self.is_trained = model_data['is_trained']
            self.training_date = model_data['training_date']
            self.metrics = model_data['metrics']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if the model supports it.
        
        Returns:
            DataFrame with feature importance or None if not supported
        """
        # Default implementation - subclasses should override if supported
        return None
    
    def predict_race_results(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the race results for all drivers.
        This higher-level method prepares features and formats predictions as race results.
        
        Args:
            input_data: DataFrame with required input data
            
        Returns:
            DataFrame with predicted race results
        """
        if not self.is_trained:
            logger.error("Model is not trained, cannot predict race results")
            return pd.DataFrame()
        
        try:
            # Prepare features
            X = self.prepare_features(input_data)
            
            # Make predictions
            predictions = self.predict(X)
            
            # Create results DataFrame
            results = pd.DataFrame()
            
            # Check if predictions are for positions or other target
            if self.target.lower() in ['position', 'finishing_position', 'race_position']:
                # Format position predictions
                results['Driver'] = input_data['Driver']
                results['PredictedPosition'] = predictions.astype(int)
                
                # Sort by predicted position
                results = results.sort_values('PredictedPosition')
                
            else:
                # For other targets (like finishing time, points, etc.)
                results['Driver'] = input_data['Driver']
                results['Predicted_' + self.target] = predictions
                
                # Sort by predictions (assuming lower is better, like time)
                results = results.sort_values('Predicted_' + self.target)
                
                # Add implied position
                results['PredictedPosition'] = range(1, len(results) + 1)
            
            # Add confidence information if available
            if hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba')):
                try:
                    probas = self.model.predict_proba(X)
                    results['Confidence'] = np.max(probas, axis=1)
                except:
                    pass
            
            logger.info(f"Predicted race results for {len(results)} drivers")
            return results
            
        except Exception as e:
            logger.error(f"Error predicting race results: {str(e)}")
            return pd.DataFrame()
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name} (v{self.version})"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        status = "Trained" if self.is_trained else "Untrained"
        train_date = self.training_date.strftime("%Y-%m-%d %H:%M") if self.training_date else "Never"
        return f"{self.name} (v{self.version}) - {status}, Trained: {train_date}"