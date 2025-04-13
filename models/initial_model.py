"""
Initial prediction model for F1 prediction project.
This model makes predictions before qualifying sessions, using only historical data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from models.base_model import F1PredictionModel

# Configure logging
logger = logging.getLogger(__name__)

class F1InitialModel(F1PredictionModel):
    """
    Initial prediction model for F1 race results.
    Makes predictions before qualifying sessions using historical data.
    """
    
    def __init__(self, estimator: str = 'rf', target: str = 'Position'):
        """
        Initialize the initial prediction model.
        
        Args:
            estimator: Type of estimator to use ('rf' for Random Forest, 'gb' for Gradient Boosting)
            target: Target variable to predict ('Position' or 'Points')
        """
        super().__init__(name='initial_model', version='0.1')
        self.estimator_type = estimator
        self.target = target
        self.feature_scaler = StandardScaler()
        
        # Initialize the model based on estimator type
        if estimator.lower() == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif estimator.lower() == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            logger.warning(f"Unknown estimator type: {estimator}, defaulting to Random Forest")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Define the features this model will use
        self.features = [
            # Driver features
            'driver_avg_points', 
            'driver_avg_positions_gained',
            'driver_finish_rate',
            'driver_form_trend',
            'driver_last3_avg_pos',
            
            # Team features
            'team_avg_points',
            'team_finish_rate',
            
            # Circuit features
            'circuit_high_speed',
            'circuit_street',
            'circuit_technical',
            'circuit_safety_car_rate',
            'overtaking_difficulty',
            
            # Weather features (if available from forecast)
            'weather_is_dry',
            'weather_is_any_wet',
            'weather_temp_mild',
            'weather_temp_hot',
            'weather_high_wind',
            
            # Driver-circuit historical performance
            'driver_circuit_avg_pos',
            
            # Interaction features
            'team_highspeed_interaction',
            'wet_driver_advantage_interaction'
        ]
        
        logger.info(f"Initialized F1InitialModel with {estimator} estimator, targeting {target}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the initial model from the input DataFrame.
        Selects relevant features and handles missing values.
        
        Args:
            df: Input DataFrame with all available data
            
        Returns:
            DataFrame with selected and prepared features
        """
        # Start with empty feature DataFrame
        X = pd.DataFrame()
        
        # Identify available features in the input data
        available_features = [f for f in self.features if f in df.columns]
        
        if not available_features:
            logger.error("None of the required features are available in the input data")
            return X
        
        logger.info(f"Found {len(available_features)} available features out of {len(self.features)} required")
        
        # Select available features
        X = df[available_features].copy()
        
        # Handle missing values - for initial model, use simple imputation
        for col in X.columns:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Fill numeric missing values with mean
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    # Fill categorical missing values with mode
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Add driver and circuit identifiers if available (for reference, not as features)
        for ref_col in ['Driver', 'TrackName', 'Circuit']:
            if ref_col in df.columns:
                X[ref_col] = df[ref_col]
        
        logger.info(f"Prepared features with shape {X.shape}")
        return X
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the initial model using historical race data.
        
        Args:
            X: Feature DataFrame
            y: Target Series with race positions or points
        """
        # Store the target type
        self.target = y.name
        
        # Keep track of the feature columns
        feature_cols = [col for col in X.columns if col not in ['Driver', 'TrackName', 'Circuit']]
        self.features = feature_cols
        
        if not feature_cols:
            logger.error("No valid feature columns found for training")
            return
        
        try:
            # Fit the model
            logger.info(f"Training {self.estimator_type} model on {len(X)} samples with {len(feature_cols)} features")
            self.model.fit(X[feature_cols], y)
            
            # Update model state
            self.is_trained = True
            self.training_date = datetime.now()
            
            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                logger.info(f"Top 5 important features: {feature_importance.head(5)['Feature'].tolist()}")
            
            logger.info(f"Model training completed successfully")
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted positions or points
        """
        if not self.is_trained:
            logger.error("Model is not trained, cannot make predictions")
            return np.array([])
        
        # Extract feature columns (excluding reference columns)
        feature_cols = [col for col in X.columns if col in self.features]
        
        if not feature_cols:
            logger.error("No valid feature columns found for prediction")
            return np.array([])
        
        try:
            # Make predictions
            predictions = self.model.predict(X[feature_cols])
            
            # For position predictions, ensure they are integers and within valid range
            if self.target.lower() in ['position', 'finishing_position', 'race_position']:
                # Round to integers
                predictions = np.round(predictions).astype(int)
                
                # Ensure positions are within valid range (1 to number of drivers)
                predictions = np.maximum(1, np.minimum(len(X), predictions))
                
                # Handle potential duplicate positions
                unique_positions = set()
                for i in range(len(predictions)):
                    while predictions[i] in unique_positions:
                        predictions[i] += 1
                    unique_positions.add(predictions[i])
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return np.array([])
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, cv: int = 3) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using grid search.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters
        """
        # Extract feature columns (excluding reference columns)
        feature_cols = [col for col in X.columns if col in self.features]
        
        if not feature_cols:
            logger.error("No valid feature columns found for hyperparameter optimization")
            return {}
        
        try:
            # Define parameter grid based on estimator type
            if self.estimator_type.lower() == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                estimator = RandomForestRegressor(random_state=42)
            elif self.estimator_type.lower() == 'gb':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                }
                estimator = GradientBoostingRegressor(random_state=42)
            else:
                logger.warning(f"Unknown estimator type: {self.estimator_type}, defaulting to Random Forest")
                param_grid = {
                    'n_estimators': [100],
                    'max_depth': [None]
                }
                estimator = RandomForestRegressor(random_state=42)
            
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('estimator', estimator)
            ])
            
            # Create grid search
            grid_search = GridSearchCV(
                pipeline,
                {'estimator__' + key: value for key, value in param_grid.items()},
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            logger.info(f"Starting hyperparameter optimization with {cv}-fold cross-validation")
            
            # Fit grid search
            grid_search.fit(X[feature_cols], y)
            
            # Get best parameters
            best_params = {key.replace('estimator__', ''): value 
                         for key, value in grid_search.best_params_.items()}
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {-grid_search.best_score_:.4f} MSE")
            
            # Update model with best parameters
            if self.estimator_type.lower() == 'rf':
                self.model = RandomForestRegressor(random_state=42, **best_params)
            elif self.estimator_type.lower() == 'gb':
                self.model = GradientBoostingRegressor(random_state=42, **best_params)
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return {}
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature importance or None if not supported
        """
        if not self.is_trained:
            logger.error("Model is not trained, cannot get feature importance")
            return None
        
        try:
            # Check if model has feature importance attribute
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                return feature_importance
            else:
                logger.warning(f"Model does not support feature importance")
                return None
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def predict_top_n(self, input_data: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """
        Predict the top N drivers for the race.
        
        Args:
            input_data: DataFrame with required input data
            n: Number of top positions to predict
            
        Returns:
            DataFrame with top N predicted drivers
        """
        # Get full race predictions
        predictions = self.predict_race_results(input_data)
        
        # Return top N
        return predictions.head(n)