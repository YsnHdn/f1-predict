"""
Pre-race prediction model for F1 prediction project.
This model makes predictions after qualifying sessions but before the race,
integrating qualifying results with historical data.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from models.base_model import F1PredictionModel
from models.initial_model import F1InitialModel

# Configure logging
logger = logging.getLogger(__name__)

class F1PreRaceModel(F1PredictionModel):
    """
    Pre-race prediction model for F1 race results.
    Makes predictions after qualifying sessions, integrating qualifying results.
    """
    
    def __init__(self, estimator: str = 'stacking', target: str = 'Position', 
                initial_model: Optional[F1InitialModel] = None):
        """
        Initialize the pre-race prediction model.
        
        Args:
            estimator: Type of estimator to use ('rf', 'gb', or 'stacking')
            target: Target variable to predict ('Position' or 'Points')
            initial_model: Optional initial model for predictions blending
        """
        super().__init__(name='pre_race_model', version='0.1')
        self.estimator_type = estimator
        self.target = target
        self.initial_model = initial_model
        self.feature_scaler = StandardScaler()
        
        # Initialize the model based on estimator type
        if estimator.lower() == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif estimator.lower() == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            )
        elif estimator.lower() == 'stacking':
            # Create a stacking ensemble
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
            ]
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5
            )
        else:
            logger.warning(f"Unknown estimator type: {estimator}, defaulting to Random Forest")
            self.model = RandomForestRegressor(n_estimators=150, random_state=42)
        
        # Define the features this model will use, including qualifying features
        self.features = [
            # Grid position features
            'GridPosition',
            'grid_front_row',
            'grid_top3',
            'grid_top5',
            'grid_top10',
            
            # Qualifying performance
            'Q1_seconds',
            'Q2_seconds',
            'Q3_seconds',
            'q_sessions_completed',
            'gap_to_pole_pct',
            
            # Driver features
            'driver_avg_points', 
            'driver_avg_positions_gained',
            'driver_finish_rate',
            'driver_form_trend',
            'driver_last3_avg_pos',
            
            # Team features
            'team_avg_points',
            'team_finish_rate',
            'grid_vs_teammate',
            
            # Circuit features
            'circuit_high_speed',
            'circuit_street',
            'circuit_technical',
            'circuit_safety_car_rate',
            'overtaking_difficulty',
            
            # Weather features
            'weather_is_dry',
            'weather_is_any_wet',
            'weather_temp_mild',
            'weather_temp_hot',
            'weather_high_wind',
            
            # Driver-circuit historical performance
            'driver_circuit_avg_pos',
            
            # Race strategy features
            'circuit_high_degradation',
            'circuit_low_degradation',
            
            # Interaction features
            'team_highspeed_interaction',
            'wet_driver_advantage_interaction',
            'grid_overtaking_interaction'
        ]
        
        logger.info(f"Initialized F1PreRaceModel with {estimator} estimator, targeting {target}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the pre-race model from the input DataFrame.
        Selects relevant features including qualifying results.
        
        Args:
            df: Input DataFrame with qualifying and historical data
            
        Returns:
            DataFrame with selected and prepared features
        """
        # Start with empty feature DataFrame
        X = pd.DataFrame()
        
        # Check if we have grid position (essential for pre-race model)
        grid_position_cols = ['GridPosition', 'Q_Position', 'QualifyingPosition', 'grid']
        has_grid_position = any(col in df.columns for col in grid_position_cols)
        
        if not has_grid_position:
            logger.error("No grid position information available, required for pre-race model")
            return X
        
        # Standardize grid position column name if needed
        for col in grid_position_cols:
            if col in df.columns:
                df['GridPosition'] = df[col]
                break
        
        # Identify available features in the input data
        available_features = [f for f in self.features if f in df.columns]
        
        # Add derived grid features if not present
        if 'GridPosition' in df.columns:
            if 'grid_front_row' not in df.columns:
                df['grid_front_row'] = (df['GridPosition'] <= 2).astype(int)
            if 'grid_top3' not in df.columns:
                df['grid_top3'] = (df['GridPosition'] <= 3).astype(int)
            if 'grid_top5' not in df.columns:
                df['grid_top5'] = (df['GridPosition'] <= 5).astype(int)
            if 'grid_top10' not in df.columns:
                df['grid_top10'] = (df['GridPosition'] <= 10).astype(int)
                
            # Add these to available features if newly created
            for feature in ['grid_front_row', 'grid_top3', 'grid_top5', 'grid_top10']:
                if feature not in available_features and feature in df.columns:
                    available_features.append(feature)
        
        if not available_features:
            logger.error("None of the required features are available in the input data")
            return X
        
        logger.info(f"Found {len(available_features)} available features out of {len(self.features)} required")
        
        # Select available features
        X = df[available_features].copy()
        
        # Handle missing values - for pre-race model, use more sophisticated imputation
        for col in X.columns:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    # If grid position related, impute based on other grid features
                    if col.startswith('grid_') or col == 'GridPosition':
                        if 'GridPosition' in X.columns and not X['GridPosition'].isna().all():
                            if col == 'GridPosition':
                                # Should not happen, but just in case
                                X[col] = X[col].fillna(X[col].median())
                            elif col == 'grid_front_row':
                                X.loc[X[col].isna(), col] = (X.loc[X[col].isna(), 'GridPosition'] <= 2).astype(int)
                            elif col == 'grid_top3':
                                X.loc[X[col].isna(), col] = (X.loc[X[col].isna(), 'GridPosition'] <= 3).astype(int)
                            elif col == 'grid_top5':
                                X.loc[X[col].isna(), col] = (X.loc[X[col].isna(), 'GridPosition'] <= 5).astype(int)
                            elif col == 'grid_top10':
                                X.loc[X[col].isna(), col] = (X.loc[X[col].isna(), 'GridPosition'] <= 10).astype(int)
                        else:
                            # Fall back to mean imputation
                            X[col] = X[col].fillna(X[col].mean())
                    # For qualifying times, impute based on grid position if possible
                    elif col.endswith('_seconds') and 'GridPosition' in X.columns:
                        # Group by grid position ranges and get mean values
                        grid_ranges = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
                        for start, end in grid_ranges:
                            mask = (X['GridPosition'] >= start) & (X['GridPosition'] <= end) & X[col].isna()
                            if mask.any():
                                # Get mean value for this grid range
                                range_mean = X.loc[(X['GridPosition'] >= start) & (X['GridPosition'] <= end), col].mean()
                                if not pd.isna(range_mean):
                                    X.loc[mask, col] = range_mean
                                else:
                                    # Fall back to overall mean
                                    X.loc[mask, col] = X[col].mean()
                    else:
                        # For other numeric features, use standard mean imputation
                        X[col] = X[col].fillna(X[col].mean())
                else:
                    # For categorical features, use mode imputation
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Add driver and circuit identifiers if available (for reference, not as features)
        for ref_col in ['Driver', 'TrackName', 'Circuit']:
            if ref_col in df.columns:
                X[ref_col] = df[ref_col]
        
        # Create interaction features if needed
        if 'GridPosition' in X.columns and 'overtaking_difficulty' in X.columns:
            if 'grid_overtaking_interaction' not in X.columns:
                X['grid_overtaking_interaction'] = X['GridPosition'] * X['overtaking_difficulty']
                available_features.append('grid_overtaking_interaction')
        
        logger.info(f"Prepared features with shape {X.shape}")
        return X
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the pre-race model using qualifying and historical race data.
        
        Args:
            X: Feature DataFrame including qualifying results
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
            X: Feature DataFrame including qualifying results
            
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
            raw_predictions = self.model.predict(X[feature_cols])
            
            # For position predictions, ensure they are integers and within valid range
            if self.target.lower() in ['position', 'finishing_position', 'race_position']:
                # Round to integers
                predictions = np.round(raw_predictions).astype(int)
                
                # Ensure positions are within valid range (1 to number of drivers)
                predictions = np.maximum(1, np.minimum(len(X), predictions))
                
                # Handle potential duplicate positions
                # Create a set to track assigned positions
                assigned_positions = set()
                
                # Create a copy of the predictions to modify
                final_predictions = predictions.copy()
                
                # Sort drivers by their predicted position (lower is better)
                indices_by_position = np.argsort(predictions)
                
                # Assign unique positions
                for idx in indices_by_position:
                    pos = int(predictions[idx])
                    
                    # Find the next available position starting from the predicted one
                    while pos in assigned_positions:
                        pos += 1
                    
                    # If we exceed the number of drivers, we need to find a lower available position
                    if pos > len(X):
                        pos = 1
                        while pos in assigned_positions:
                            pos += 1
                    
                    # Assign the position and mark it as taken
                    final_predictions[idx] = pos
                    assigned_positions.add(pos)
                
                # Verify all positions are unique and within range
                assert len(assigned_positions) == len(X), "Failed to assign unique positions to all drivers"
                assert max(assigned_positions) <= len(X), "Assigned position exceeds number of drivers"
                assert min(assigned_positions) >= 1, "Assigned position is less than 1"
                
                return final_predictions
            else:
                # For other targets (like points), return raw predictions
                return raw_predictions
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return np.array([])
    
    def blend_with_initial_model(self, X: pd.DataFrame, blend_weight: float = 0.7) -> pd.DataFrame:
        """
        Blend predictions with the initial model for more robust results.
        
        Args:
            X: Feature DataFrame including qualifying results
            blend_weight: Weight to give to the pre-race model (0-1)
            
        Returns:
            DataFrame with blended predictions
        """
        if not self.is_trained:
            logger.error("Pre-race model is not trained, cannot blend predictions")
            return pd.DataFrame()
        
        if self.initial_model is None or not self.initial_model.is_trained:
            logger.warning("Initial model not available or not trained, using pre-race model only")
            return self.predict_race_results(X)
        
        try:
            # Get predictions from both models
            pre_race_predictions = self.predict_race_results(X)
            initial_predictions = self.initial_model.predict_race_results(X)
            
            if pre_race_predictions.empty or initial_predictions.empty:
                logger.error("One or both models failed to generate predictions")
                return pre_race_predictions if not pre_race_predictions.empty else initial_predictions
            
            # Ensure both prediction sets have the same drivers
            common_drivers = set(pre_race_predictions['Driver']).intersection(set(initial_predictions['Driver']))
            
            if not common_drivers:
                logger.error("No common drivers found in both prediction sets")
                return pre_race_predictions
            
            # Filter for common drivers
            pre_race_predictions = pre_race_predictions[pre_race_predictions['Driver'].isin(common_drivers)]
            initial_predictions = initial_predictions[initial_predictions['Driver'].isin(common_drivers)]
            
            # Create a new DataFrame for blended predictions
            blended_predictions = pre_race_predictions.copy()
            
            # Blend the position predictions
            if 'PredictedPosition' in pre_race_predictions.columns and 'PredictedPosition' in initial_predictions.columns:
                # Create mapping from driver to initial model position
                initial_positions = {row['Driver']: row['PredictedPosition'] 
                                   for _, row in initial_predictions.iterrows()}
                
                # Blend positions
                for i, row in blended_predictions.iterrows():
                    driver = row['Driver']
                    pre_race_pos = row['PredictedPosition']
                    initial_pos = initial_positions.get(driver, pre_race_pos)
                    
                    # Weighted blend, with some randomization to avoid ties
                    blended_pos = (pre_race_pos * blend_weight) + (initial_pos * (1 - blend_weight))
                    blended_predictions.loc[i, 'BlendedPosition'] = blended_pos
                
                # Sort by blended position
                blended_predictions = blended_predictions.sort_values('BlendedPosition')
                
                # Reassign integer positions
                blended_predictions['PredictedPosition'] = range(1, len(blended_predictions) + 1)
                
                # Drop the temporary column
                blended_predictions = blended_predictions.drop('BlendedPosition', axis=1)
            
            logger.info(f"Blended predictions from both models with weight {blend_weight} for pre-race model")
            return blended_predictions
            
        except Exception as e:
            logger.error(f"Error blending predictions: {str(e)}")
            return self.predict_race_results(X)
    
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
                    'n_estimators': [100, 150, 200],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                estimator = RandomForestRegressor(random_state=42)
            elif self.estimator_type.lower() == 'gb':
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [4, 6, 8],
                    'min_samples_split': [2, 5, 10]
                }
                estimator = GradientBoostingRegressor(random_state=42)
            elif self.estimator_type.lower() == 'stacking':
                # For stacking, optimize the meta-estimator and the base estimators
                param_grid = {
                    'final_estimator__alpha': [0.1, 1.0, 10.0],
                    'estimators__rf__n_estimators': [100, 150],
                    'estimators__gb__n_estimators': [100, 150]
                }
                estimators = [
                    ('rf', RandomForestRegressor(random_state=42)),
                    ('gb', GradientBoostingRegressor(random_state=42))
                ]
                estimator = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(),
                    cv=5
                )
            else:
                logger.warning(f"Unknown estimator type: {self.estimator_type}, defaulting to Random Forest")
                param_grid = {
                    'n_estimators': [150],
                    'max_depth': [20]
                }
                estimator = RandomForestRegressor(random_state=42)
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator,
                param_grid,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            logger.info(f"Starting hyperparameter optimization with {cv}-fold cross-validation")
            
            # Fit grid search
            grid_search.fit(X[feature_cols], y)
            
            # Get best parameters
            best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best score: {-grid_search.best_score_:.4f} MSE")
            
            # Update model with best parameters
            if self.estimator_type.lower() == 'rf':
                self.model = RandomForestRegressor(random_state=42, **best_params)
            elif self.estimator_type.lower() == 'gb':
                self.model = GradientBoostingRegressor(random_state=42, **best_params)
            elif self.estimator_type.lower() == 'stacking':
                # Extract parameters for each component
                rf_params = {k.replace('estimators__rf__', ''): v for k, v in best_params.items() 
                          if k.startswith('estimators__rf__')}
                gb_params = {k.replace('estimators__gb__', ''): v for k, v in best_params.items() 
                          if k.startswith('estimators__gb__')}
                ridge_params = {k.replace('final_estimator__', ''): v for k, v in best_params.items() 
                             if k.startswith('final_estimator__')}
                
                # Create optimized stacking ensemble
                estimators = [
                    ('rf', RandomForestRegressor(random_state=42, **rf_params)),
                    ('gb', GradientBoostingRegressor(random_state=42, **gb_params))
                ]
                self.model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=Ridge(**ridge_params),
                    cv=5
                )
            
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
            # Direct feature importance for RF and GB
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                return feature_importance
            
            # For stacking ensemble, get from the base models
            elif self.estimator_type.lower() == 'stacking':
                importances = {}
                total_features = len(self.features)
                
                # Try to get importances from base estimators
                for name, estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        for i, imp in enumerate(estimator.feature_importances_):
                            if i < total_features:
                                feature = self.features[i]
                                importances[feature] = importances.get(feature, 0) + imp / len(self.model.estimators_)
                
                if importances:
                    feature_importance = pd.DataFrame({
                        'Feature': list(importances.keys()),
                        'Importance': list(importances.values())
                    }).sort_values('Importance', ascending=False)
                    
                    return feature_importance
            
            logger.warning(f"Model does not support feature importance retrieval")
            return None
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def predict_podium(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the podium (top 3) finishers for the race.
        
        Args:
            input_data: DataFrame with required input data
            
        Returns:
            DataFrame with predicted podium drivers
        """
        # Get full race predictions
        predictions = self.predict_race_results(input_data)
        
        # Return top 3
        return predictions.head(3)