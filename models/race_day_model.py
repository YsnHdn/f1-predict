"""
Race day prediction model for F1 prediction project.
This model makes final predictions on race day, integrating last-minute information
such as up-to-date weather, grid penalties, tire choices, and pre-race events.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from models.base_model import F1PredictionModel
from models.initial_model import F1InitialModel
from models.pre_race_model import F1PreRaceModel

# Configure logging
logger = logging.getLogger(__name__)

class F1RaceDayModel(F1PredictionModel):
    """
    Race day prediction model for F1 race results.
    Makes final predictions just before the race start,
    incorporating last-minute information and blending predictions from previous models.
    """
    
    def __init__(self, estimator: str = 'voting', target: str = 'Position', 
                pre_race_model: Optional[F1PreRaceModel] = None,
                initial_model: Optional[F1InitialModel] = None):
        """
        Initialize the race day prediction model.
        
        Args:
            estimator: Type of estimator to use ('rf', 'gb', or 'voting')
            target: Target variable to predict ('Position' or 'Points')
            pre_race_model: Optional pre-race model for predictions blending
            initial_model: Optional initial model for predictions blending
        """
        super().__init__(name='race_day_model', version='0.1')
        self.estimator_type = estimator
        self.target = target
        self.pre_race_model = pre_race_model
        self.initial_model = initial_model
        self.feature_scaler = StandardScaler()
        
        # Initialize the model based on estimator type
        if estimator.lower() == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=25,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            )
        elif estimator.lower() == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                random_state=42
            )
        elif estimator.lower() == 'voting':
            # Create a voting ensemble
            estimators = [
                ('rf', RandomForestRegressor(n_estimators=150, max_depth=25, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=7, random_state=42))
            ]
            self.model = VotingRegressor(
                estimators=estimators,
                weights=[1, 1]
            )
        else:
            logger.warning(f"Unknown estimator type: {estimator}, defaulting to Random Forest")
            self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        
        # Define the features this model will use, including race day specific features
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
            
            # Weather features - updated race day forecast
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
            'grid_overtaking_interaction',
            
            # Race day specific features
            'starting_tire_compound',  # Tire compound chosen for race start
            'grid_penalty_applied',    # Whether the driver received a grid penalty
            'original_grid_position',  # Grid position before penalties
            'formation_lap_issue',     # Any issues during formation lap
            'track_temp_celsius',      # Track temperature
            'track_evolution',         # Expected track evolution (0-10 scale)
            'pit_lane_position',       # Position in pit lane (advantage for undercuts)
            'team_recent_strategy',    # Team's strategy in recent similar races
            'recent_race_pace',        # Race pace in recent practice sessions
            'car_upgrades_applied',    # Significant upgrades since last race
        ]
        
        logger.info(f"Initialized F1RaceDayModel with {estimator} estimator, targeting {target}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the race day model from the input DataFrame.
        Selects relevant features including last-minute information.
        
        Args:
            df: Input DataFrame with qualifying and historical data
            
        Returns:
            DataFrame with selected and prepared features
        """
        # Start with empty feature DataFrame
        X = pd.DataFrame()
        
        # Check if we have grid position (essential for race day model)
        grid_position_cols = ['GridPosition', 'Q_Position', 'QualifyingPosition', 'grid']
        has_grid_position = any(col in df.columns for col in grid_position_cols)
        
        if not has_grid_position:
            logger.error("No grid position information available, required for race day model")
            return X
        
        # Standardize grid position column name if needed
        for col in grid_position_cols:
            if col in df.columns:
                df['GridPosition'] = df[col]
                break
        
        # Process last-minute grid changes (penalties, etc.)
        if 'GridPenalty' in df.columns:
            # Store original grid position before penalty
            if 'original_grid_position' not in df.columns:
                df['original_grid_position'] = df['GridPosition']
                
            # Apply penalties to grid positions
            df['GridPosition'] = df['GridPosition'] + df['GridPenalty']
            
            # Flag drivers with penalties
            df['grid_penalty_applied'] = (df['GridPenalty'] > 0).astype(int)
        
        # Process tire compound choices if available
        if 'StartingTireCompound' in df.columns:
            # One-hot encode tire compounds
            tire_compounds = ['soft', 'medium', 'hard', 'intermediate', 'wet']
            for compound in tire_compounds:
                df[f'starting_tire_{compound}'] = (df['StartingTireCompound'].str.lower() == compound).astype(int)
            
            # Drop the original string column as it's not usable for ML models
            df = df.drop('StartingTireCompound', axis=1)
            
        # Similarly, if we have string value for starting_tire_compound, encode it
        if 'starting_tire_compound' in df.columns and pd.api.types.is_string_dtype(df['starting_tire_compound']):
            # One-hot encode tire compounds
            tire_compounds = ['soft', 'medium', 'hard', 'intermediate', 'wet']
            for compound in tire_compounds:
                df[f'starting_tire_{compound}'] = (df['starting_tire_compound'].str.lower() == compound).astype(int)
            
            # Drop the original string column
            df = df.drop('starting_tire_compound', axis=1)
        
        # Update grid-related features after any changes
        if 'GridPosition' in df.columns:
            if 'grid_front_row' not in df.columns:
                df['grid_front_row'] = (df['GridPosition'] <= 2).astype(int)
            if 'grid_top3' not in df.columns:
                df['grid_top3'] = (df['GridPosition'] <= 3).astype(int)
            if 'grid_top5' not in df.columns:
                df['grid_top5'] = (df['GridPosition'] <= 5).astype(int)
            if 'grid_top10' not in df.columns:
                df['grid_top10'] = (df['GridPosition'] <= 10).astype(int)
        
        # Update weather-related features with last-minute forecast
        if 'RaceWeatherCondition' in df.columns:
            weather_map = {
                'dry': 'dry',
                'damp': 'damp', 
                'light rain': 'wet',
                'rain': 'wet', 
                'heavy rain': 'very_wet',
                'storm': 'very_wet'
            }
            # Map to standardized weather condition
            df['weather_racing_condition'] = df['RaceWeatherCondition'].str.lower().map(
                lambda x: next((v for k, v in weather_map.items() if k in x), 'dry')
            )
            
            # Update binary indicators
            df['weather_is_dry'] = (df['weather_racing_condition'] == 'dry').astype(int)
            df['weather_is_any_wet'] = (~df['weather_is_dry']).astype(int)
        
        # Identify available features in the input data
        available_features = [f for f in self.features if f in df.columns]
        
        if not available_features:
            logger.error("None of the required features are available in the input data")
            return X
        
        logger.info(f"Found {len(available_features)} available features out of {len(self.features)} required")
        
        # Select available features
        X = df[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    # For numeric features, use mean imputation
                    X[col] = X[col].fillna(X[col].mean())
                else:
                    # For categorical features, use mode imputation
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Add driver and circuit identifiers if available (for reference, not as features)
        for ref_col in ['Driver', 'TrackName', 'Circuit']:
            if ref_col in df.columns:
                X[ref_col] = df[ref_col]
        
        # Create or update interaction features
        # Grid position × Weather interaction (wet conditions can benefit lower grid positions)
        if 'GridPosition' in X.columns and 'weather_is_any_wet' in X.columns:
            X['grid_weather_interaction'] = X['GridPosition'] * X['weather_is_any_wet']
        
        # Driver wet advantage × Current weather
        if 'driver_wet_advantage' in X.columns and 'weather_is_any_wet' in X.columns:
            X['current_wet_advantage'] = X['driver_wet_advantage'] * X['weather_is_any_wet']
        
        # Tire compound × Track temperature interaction
        if 'starting_tire_soft' in X.columns and 'track_temp_celsius' in X.columns:
            X['soft_temp_interaction'] = X['starting_tire_soft'] * X['track_temp_celsius']
        
        logger.info(f"Prepared features with shape {X.shape}")
        return X
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the race day model using qualifying, historical and race day data.
        
        Args:
            X: Feature DataFrame including race day information
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
            self._log_feature_importance()
            
            logger.info(f"Model training completed successfully")
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
    
    def _log_feature_importance(self) -> None:
        """Log the feature importance if available."""
        try:
            # Try to get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                logger.info(f"Top 5 important features: {feature_importance.head(5)['Feature'].tolist()}")
            elif self.estimator_type.lower() == 'voting':
                # Try to get from first estimator
                if hasattr(self.model.estimators_[0], 'feature_importances_'):
                    importances = self.model.estimators_[0].feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': self.features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    logger.info(f"Top 5 important features from first estimator: {feature_importance.head(5)['Feature'].tolist()}")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {str(e)}")
    
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
    
    def ensemble_predict(self, X: pd.DataFrame, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Make ensemble predictions by combining this model with pre-race and initial models.
        
        Args:
            X: Feature DataFrame including race day information
            weights: Optional dictionary of weights for each model ('race_day', 'pre_race', 'initial')
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not self.is_trained:
            logger.error("Race day model is not trained, cannot make ensemble predictions")
            return pd.DataFrame()
        
        # Set default weights if not provided
        if weights is None:
            weights = {'race_day': 0.6, 'pre_race': 0.3, 'initial': 0.1}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Check which models are available
        available_models = {'race_day': self}
        if self.pre_race_model is not None and self.pre_race_model.is_trained:
            available_models['pre_race'] = self.pre_race_model
        if self.initial_model is not None and self.initial_model.is_trained:
            available_models['initial'] = self.initial_model
        
        if len(available_models) == 1:
            logger.warning("Only race day model available, using standalone predictions")
            return self.predict_race_results(X)
        
        try:
            # Get predictions from each available model
            predictions = {}
            for model_name, model in available_models.items():
                preds = model.predict_race_results(X)
                if not preds.empty:
                    predictions[model_name] = preds
            
            if not predictions:
                logger.error("No valid predictions generated by any model")
                return pd.DataFrame()
            
            # Get list of drivers from the race day model predictions
            base_predictions = predictions['race_day']
            drivers = base_predictions['Driver'].tolist()
            
            # Create a new DataFrame for ensemble predictions
            ensemble_df = pd.DataFrame({'Driver': drivers})
            
            # Calculate weighted position scores
            ensemble_df['WeightedPositionScore'] = 0
            
            for model_name, preds in predictions.items():
                if 'PredictedPosition' in preds.columns:
                    # Create driver to position mapping
                    pos_map = {row['Driver']: row['PredictedPosition'] for _, row in preds.iterrows()}
                    
                    # Apply weight to each model's predictions
                    weight = weights.get(model_name, 0)
                    for driver in drivers:
                        if driver in pos_map:
                            ensemble_df.loc[ensemble_df['Driver'] == driver, 'WeightedPositionScore'] += pos_map[driver] * weight
            
            # Sort by weighted position score
            ensemble_df = ensemble_df.sort_values('WeightedPositionScore')
            
            # Assign final positions
            ensemble_df['PredictedPosition'] = range(1, len(ensemble_df) + 1)
            
            # Add confidence information if available
            confidence_available = False
            for model_name, preds in predictions.items():
                if 'Confidence' in preds.columns:
                    confidence_map = {row['Driver']: row['Confidence'] for _, row in preds.iterrows()}
                    weight = weights.get(model_name, 0)
                    
                    if 'WeightedConfidence' not in ensemble_df.columns:
                        ensemble_df['WeightedConfidence'] = 0
                        
                    for driver in drivers:
                        if driver in confidence_map:
                            ensemble_df.loc[ensemble_df['Driver'] == driver, 'WeightedConfidence'] += confidence_map[driver] * weight
                    
                    confidence_available = True
            
            if confidence_available:
                # Normalize confidence scores
                max_conf = ensemble_df['WeightedConfidence'].max()
                if max_conf > 0:
                    ensemble_df['Confidence'] = ensemble_df['WeightedConfidence'] / max_conf
                
                # Drop temporary column
                ensemble_df = ensemble_df.drop('WeightedConfidence', axis=1)
            
            # Drop temporary column
            ensemble_df = ensemble_df.drop('WeightedPositionScore', axis=1)
            
            logger.info(f"Generated ensemble predictions using {len(predictions)} models")
            return ensemble_df
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
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
                    'n_estimators': [150, 200, 250],
                    'max_depth': [20, 25, 30, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 3]
                }
                estimator = RandomForestRegressor(random_state=42)
            elif self.estimator_type.lower() == 'gb':
                param_grid = {
                    'n_estimators': [150, 200, 250],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [5, 7, 9],
                    'min_samples_split': [2, 4, 6]
                }
                estimator = GradientBoostingRegressor(random_state=42)
            elif self.estimator_type.lower() == 'voting':
                # For voting ensemble, optimize weights
                param_grid = {
                    'weights': [[1, 1], [1, 2], [2, 1], [3, 2], [2, 3]]
                }
                estimators = [
                    ('rf', RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42))
                ]
                estimator = VotingRegressor(estimators=estimators)
            else:
                logger.warning(f"Unknown estimator type: {self.estimator_type}, defaulting to Random Forest")
                param_grid = {
                    'n_estimators': [200],
                    'max_depth': [25]
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
            elif self.estimator_type.lower() == 'voting':
                estimators = [
                    ('rf', RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42))
                ]
                self.model = VotingRegressor(
                    estimators=estimators,
                    weights=best_params['weights']
                )
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}")
            return {}
    
    def predict_with_scenarios(self, X: pd.DataFrame, scenarios: List[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """
        Make predictions under different race scenarios (e.g., dry race, wet race, safety car).
        
        Args:
            X: Feature DataFrame with race day information
            scenarios: List of dictionaries with scenario parameters
            
        Returns:
            Dictionary mapping scenario names to prediction DataFrames
        """
        if not self.is_trained:
            logger.error("Model is not trained, cannot make scenario predictions")
            return {}
        
        # Define default scenarios if none provided
        if scenarios is None:
            scenarios = [
                {
                    'name': 'Baseline',
                    'description': 'Current conditions, no changes',
                    'parameters': {}
                },
                {
                    'name': 'Wet Race',
                    'description': 'Rain during race',
                    'parameters': {
                        'weather_is_dry': 0,
                        'weather_is_any_wet': 1,
                        'weather_racing_condition': 'wet'
                    }
                },
                {
                    'name': 'Safety Car',
                    'description': 'Race with safety car intervention',
                    'parameters': {
                        'safety_car_deployed': 1
                    }
                },
                {
                    'name': 'Hot Race',
                    'description': 'Higher than expected temperatures',
                    'parameters': {
                        'weather_temp_hot': 1,
                        'weather_temp_mild': 0,
                        'track_temp_celsius': lambda x: x * 1.2 if 'track_temp_celsius' in x.columns else 45
                    }
                }
            ]
        
        # Prepare features for the base scenario
        base_features = self.prepare_features(X)
        
        # Generate predictions for each scenario
        scenario_predictions = {}
        
        for scenario in scenarios:
            try:
                scenario_name = scenario.get('name', f"Scenario_{len(scenario_predictions) + 1}")
                logger.info(f"Generating predictions for scenario: {scenario_name}")
                
                # Create a copy of the base features
                scenario_X = base_features.copy()
                
                # Apply scenario parameters
                for param, value in scenario.get('parameters', {}).items():
                    if param in scenario_X.columns:
                        # Apply callable parameters (functions)
                        if callable(value):
                            scenario_X[param] = value(scenario_X)
                        else:
                            # Apply direct value assignment
                            scenario_X[param] = value
                
                # Make predictions
                if self.initial_model is not None and self.pre_race_model is not None:
                    # Use ensemble prediction if available
                    predictions = self.ensemble_predict(scenario_X)
                else:
                    # Use standalone prediction
                    predictions = self.predict_race_results(scenario_X)
                
                # Add scenario information
                if not predictions.empty:
                    predictions['Scenario'] = scenario_name
                    predictions['ScenarioDescription'] = scenario.get('description', '')
                
                # Store predictions
                scenario_predictions[scenario_name] = predictions
                
            except Exception as e:
                logger.error(f"Error generating predictions for scenario {scenario.get('name', 'unknown')}: {str(e)}")
        
        return scenario_predictions
    
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
            
            # For voting ensemble, get from the base models
            elif self.estimator_type.lower() == 'voting':
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