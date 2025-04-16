"""
Performance analyzer agent for F1 prediction project.
This agent analyzes the accuracy of predictions compared to actual race results.
"""

import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from agents.base_agent import F1BaseAgent
from agents.utils.logging import AgentLogger
from models.evaluation.metrics import calculate_prediction_metrics

# Configure logging
logger = logging.getLogger(__name__)

class PerformanceAnalyzer(F1BaseAgent):
    """
    Agent responsible for analyzing prediction performance.
    Compares predictions to actual race results and computes
    accuracy metrics to evaluate model performance.
    """
    
    def __init__(self, output_dir: str = "analysis"):
        """
        Initialize the performance analyzer agent.
        
        Args:
            output_dir: Directory to store analysis results
        """
        super().__init__(
            name="Performance Analyzer",
            description="Analyzes the accuracy of Formula 1 race predictions",
            goal="Evaluate prediction models and provide insights for improvement"
        )
        
        self.output_dir = output_dir
        self.agent_logger = AgentLogger(agent_name="PerformanceAnalyzer")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create subdirectories
        for subdir in ['metrics', 'charts', 'reports']:
            path = os.path.join(output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
        
        self.agent_logger.info(f"Initialized PerformanceAnalyzer with output directory: {output_dir}")
    
    def get_backstory(self) -> str:
        """
        Get the agent's backstory for CrewAI.
        
        Returns:
            String containing the agent's backstory
        """
        return (
            "I am a seasoned data analyst with expertise in evaluating predictive models. "
            "With a background in both statistics and motorsport analytics, I specialize in "
            "assessing the accuracy of Formula 1 race predictions. I can identify patterns in "
            "prediction errors, understand where models excel or struggle, and provide actionable "
            "insights to improve future predictions. My goal is to continuously refine our "
            "prediction capabilities through rigorous analysis and feedback."
        )
    
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the performance analysis process.
        
        Args:
            context: Context with information needed for analysis
                Expected keys:
                - race_info: Information about the race
                - data_paths: Paths to data files
                - predictions: Dictionary with prediction results
                
        Returns:
            Dictionary with analysis results
        """
        if context is None:
            context = {}
        
        # Extract parameters from context
        race_info = context.get('race_info', {})
        data_paths = context.get('data_paths', {})
        predictions = context.get('predictions', {})
        
        # Validate inputs
        if not race_info:
            self.agent_logger.error("No race information provided for analysis")
            raise ValueError("Race information is required for analysis")
        
        if not data_paths:
            self.agent_logger.error("No data paths provided for analysis")
            raise ValueError("Data paths are required for analysis")
        
        if not predictions:
            self.agent_logger.error("No predictions provided for analysis")
            raise ValueError("Predictions are required for analysis")
        
        race_name = race_info.get('name', 'Unknown Race')
        self.agent_logger.info(f"Starting performance analysis for {race_name}")
        
        # Store results
        results = {
            'race_info': race_info,
            'metrics': {},
            'charts': {},
            'reports': {},
            'summary': {}
        }
        
        try:
            # Load actual race results
            actual_results = None
            
            # Look for race results in data_paths
            for key, path in data_paths.items():
                if '_R' in key:
                    try:
                        actual_results = pd.read_csv(path)
                        self.agent_logger.info(f"Loaded actual race results from {path}")
                        break
                    except Exception as e:
                        self.agent_logger.error(f"Error loading race results from {path}: {str(e)}")
            
            if actual_results is None or actual_results.empty:
                self.agent_logger.error("Could not find actual race results")
                raise ValueError("Actual race results are required for analysis")
            
            # Check if actual_results has the required columns
            required_columns = ['Driver', 'Position']
            if not all(col in actual_results.columns for col in required_columns):
                self.agent_logger.error(f"Actual results missing required columns: {required_columns}")
                raise ValueError("Actual results must contain Driver and Position columns")
            
            # Analyze each prediction type
            for pred_type, pred_results in predictions.items():
                self.agent_logger.task_start(f"Analyzing {pred_type} prediction")
                
                # Extract prediction DataFrame from results
                if isinstance(pred_results, dict) and 'predictions' in pred_results:
                    pred_df = pred_results['predictions']
                elif isinstance(pred_results, str) and os.path.exists(pred_results):
                    # If it's a file path, load it
                    pred_df = pd.read_csv(pred_results)
                elif isinstance(pred_results, pd.DataFrame):
                    pred_df = pred_results
                else:
                    self.agent_logger.warning(f"Could not extract predictions from {pred_type} results")
                    continue
                
                # If predictions is empty, skip
                if pred_df is None or pred_df.empty:
                    self.agent_logger.warning(f"No predictions available for {pred_type}")
                    continue
                
                # Check if pred_df has the required columns
                if not all(col in pred_df.columns for col in ['Driver', 'PredictedPosition']):
                    self.agent_logger.warning(f"Predictions missing required columns: Driver, PredictedPosition")
                    continue
                
                # Merge predictions with actual results
                analysis_df = pd.merge(
                    actual_results[['Driver', 'Position']], 
                    pred_df[['Driver', 'PredictedPosition']],
                    on='Driver',
                    how='inner'
                )
                
                # If no common drivers, skip
                if analysis_df.empty:
                    self.agent_logger.warning(f"No common drivers between predictions and actual results")
                    continue
                
                # Calculate metrics
                metrics = self._calculate_metrics(analysis_df)
                results['metrics'][pred_type] = metrics
                
                # Generate charts
                charts = self._generate_charts(analysis_df, pred_type, race_info)
                results['charts'][pred_type] = charts
                
                # Generate report
                report = self._generate_report(analysis_df, metrics, pred_type, race_info)
                results['reports'][pred_type] = report
                
                self.agent_logger.task_complete(f"Analyzing {pred_type} prediction")
            
            # Compile overall summary
            summary = self._compile_summary(results)
            results['summary'] = summary
            
            # Publish analysis completed event
            self.publish_event("analysis_completed", {
                "race_name": race_name,
                "metrics_summary": summary.get('metrics_summary', {}),
                "best_prediction_type": summary.get('best_prediction', None)
            })
            
            return results
            
        except Exception as e:
            self.agent_logger.error(f"Error during performance analysis: {str(e)}")
            
            # Publish analysis failed event
            self.publish_event("analysis_failed", {
                "race_name": race_name,
                "error": str(e)
            })
            
            raise
    
    def _calculate_metrics(self, analysis_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            analysis_df: DataFrame with actual and predicted positions
            
        Returns:
            Dictionary with calculated metrics
        """
        # Calculate standard metrics
        mae = mean_absolute_error(analysis_df['Position'], analysis_df['PredictedPosition'])
        rmse = np.sqrt(mean_squared_error(analysis_df['Position'], analysis_df['PredictedPosition']))
        
        # Calculate position-specific metrics
        exact_matches = (analysis_df['Position'] == analysis_df['PredictedPosition']).mean()
        within_one = (abs(analysis_df['Position'] - analysis_df['PredictedPosition']) <= 1).mean()
        within_three = (abs(analysis_df['Position'] - analysis_df['PredictedPosition']) <= 3).mean()
        
        # Calculate top-N accuracy
        top1_actual = analysis_df[analysis_df['Position'] == 1]['Driver'].values
        top1_predicted = analysis_df[analysis_df['PredictedPosition'] == 1]['Driver'].values
        top1_accuracy = len(set(top1_actual) & set(top1_predicted)) / len(top1_actual) if len(top1_actual) > 0 else 0
        
        top3_actual = set(analysis_df[analysis_df['Position'] <= 3]['Driver'].values)
        top3_predicted = set(analysis_df[analysis_df['PredictedPosition'] <= 3]['Driver'].values)
        top3_accuracy = len(top3_actual & top3_predicted) / len(top3_actual) if len(top3_actual) > 0 else 0
        
        # Calculate podium order accuracy
        podium_correct = False
        if len(top3_actual) >= 3 and len(top3_predicted) >= 3:
            actual_podium = analysis_df[analysis_df['Position'] <= 3].sort_values('Position')['Driver'].values
            predicted_podium = analysis_df[analysis_df['PredictedPosition'] <= 3].sort_values('PredictedPosition')['Driver'].values
            podium_correct = np.array_equal(actual_podium, predicted_podium)
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'exact_position_accuracy': exact_matches,
            'within_one_accuracy': within_one,
            'within_three_accuracy': within_three,
            'winner_prediction_accuracy': top1_accuracy,
            'podium_prediction_accuracy': top3_accuracy,
            'podium_order_correct': podium_correct
        }
        
        return metrics
    
    def _generate_charts(self, analysis_df: pd.DataFrame, pred_type: str, 
                       race_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations of prediction accuracy.
        
        Args:
            analysis_df: DataFrame with actual and predicted positions
            pred_type: Type of prediction being analyzed
            race_info: Information about the race
            
        Returns:
            Dictionary with paths to generated charts
        """
        charts = {}
        race_name = race_info.get('name', 'Unknown').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Actual vs. Predicted positions scatter plot
            fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
            
            # Plot perfect prediction line
            max_pos = max(analysis_df['Position'].max(), analysis_df['PredictedPosition'].max())
            ax.plot([1, max_pos], [1, max_pos], 'k--', alpha=0.5, label='Perfect Prediction')
            
            # Plot actual vs predicted
            sns.scatterplot(x='Position', y='PredictedPosition', data=analysis_df, ax=ax, s=100)
            
            # Add driver labels to points
            for idx, row in analysis_df.iterrows():
                ax.annotate(row['Driver'], (row['Position'], row['PredictedPosition']), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Actual Position')
            ax.set_ylabel('Predicted Position')
            ax.set_title(f'{race_info.get("name", "Race")} - {pred_type.title()} Prediction Accuracy')
            
            # Set integer ticks
            ax.set_xticks(range(1, max_pos + 1))
            ax.set_yticks(range(1, max_pos + 1))
            
            # Save figure
            scatter_path = os.path.join(self.output_dir, 'charts', 
                                      f"{timestamp}_{race_name}_{pred_type}_scatter.png")
            plt.savefig(scatter_path)
            plt.close()
            
            charts['position_scatter'] = scatter_path
            
            # 2. Position Error bar chart
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            # Calculate position error
            analysis_df['PositionError'] = analysis_df['PredictedPosition'] - analysis_df['Position']
            
            # Sort by actual position
            sorted_df = analysis_df.sort_values('Position')
            
            # Plot position error
            bar_plot = sns.barplot(x='Driver', y='PositionError', data=sorted_df, ax=ax)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Rotate x labels
            plt.xticks(rotation=45, ha='right')
            
            ax.set_ylabel('Position Error (Predicted - Actual)')
            ax.set_title(f'{race_info.get("name", "Race")} - {pred_type.title()} Position Error by Driver')
            
            # Save figure
            error_path = os.path.join(self.output_dir, 'charts', 
                                    f"{timestamp}_{race_name}_{pred_type}_error.png")
            plt.savefig(error_path, bbox_inches='tight')
            plt.close()
            
            charts['position_error'] = error_path
            
            # 3. Heatmap of actual vs predicted positions
            fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
            
            # Create position matrix
            max_pos = max(analysis_df['Position'].max(), analysis_df['PredictedPosition'].max())
            position_matrix = np.zeros((max_pos, max_pos))
            
            for idx, row in analysis_df.iterrows():
                actual_pos = int(row['Position']) - 1  # 0-indexed
                pred_pos = int(row['PredictedPosition']) - 1  # 0-indexed
                position_matrix[actual_pos, pred_pos] += 1
            
            # Plot heatmap
            sns.heatmap(position_matrix, annot=True, fmt='g', cmap='viridis', ax=ax)
            
            ax.set_xlabel('Predicted Position')
            ax.set_ylabel('Actual Position')
            ax.set_title(f'{race_info.get("name", "Race")} - {pred_type.title()} Position Distribution')
            
            # Set labels (1-indexed)
            ax.set_xticks(np.arange(max_pos) + 0.5)
            ax.set_yticks(np.arange(max_pos) + 0.5)
            ax.set_xticklabels(range(1, max_pos + 1))
            ax.set_yticklabels(range(1, max_pos + 1))
            
            # Save figure
            heatmap_path = os.path.join(self.output_dir, 'charts', 
                                      f"{timestamp}_{race_name}_{pred_type}_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()
            
            charts['position_heatmap'] = heatmap_path
            
            return charts
            
        except Exception as e:
            self.agent_logger.error(f"Error generating charts: {str(e)}")
            return charts
    
    def _generate_report(self, analysis_df: pd.DataFrame, metrics: Dict[str, float], 
                       pred_type: str, race_info: Dict[str, Any]) -> str:
        """
        Generate a detailed report of prediction accuracy.
        
        Args:
            analysis_df: DataFrame with actual and predicted positions
            metrics: Dictionary with calculated metrics
            pred_type: Type of prediction being analyzed
            race_info: Information about the race
            
        Returns:
            Path to the generated report file
        """
        race_name = race_info.get('name', 'Unknown').replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report filename
        report_path = os.path.join(self.output_dir, 'reports', 
                                 f"{timestamp}_{race_name}_{pred_type}_report.md")
        
        try:
            # Build report content
            report_content = f"""# {race_info.get('name', 'Race')} - {pred_type.title()} Prediction Analysis
            
## Summary

- **Date**: {race_info.get('date', 'Unknown')}
- **Circuit**: {race_info.get('circuit', 'Unknown')}
- **Prediction Type**: {pred_type.title()}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

- **Mean Absolute Error (MAE)**: {metrics['mae']:.2f}
- **Root Mean Squared Error (RMSE)**: {metrics['rmse']:.2f}
- **Exact Position Accuracy**: {metrics['exact_position_accuracy']:.2f} ({metrics['exact_position_accuracy']*100:.1f}%)
- **Within 1 Position Accuracy**: {metrics['within_one_accuracy']:.2f} ({metrics['within_one_accuracy']*100:.1f}%)
- **Within 3 Positions Accuracy**: {metrics['within_three_accuracy']:.2f} ({metrics['within_three_accuracy']*100:.1f}%)
- **Winner Prediction Accuracy**: {metrics['winner_prediction_accuracy']:.2f} ({metrics['winner_prediction_accuracy']*100:.1f}%)
- **Podium Prediction Accuracy**: {metrics['podium_prediction_accuracy']:.2f} ({metrics['podium_prediction_accuracy']*100:.1f}%)
- **Podium Order Correct**: {'Yes' if metrics['podium_order_correct'] else 'No'}

## Detailed Results

| Driver | Actual Position | Predicted Position | Error |
|--------|----------------|-------------------|-------|
"""
            
            # Add driver-by-driver results
            sorted_results = analysis_df.sort_values('Position')
            for idx, row in sorted_results.iterrows():
                report_content += f"| {row['Driver']} | {int(row['Position'])} | {int(row['PredictedPosition'])} | {int(row['PredictedPosition']) - int(row['Position'])} |\n"
            
            # Add notes and observations
            report_content += """
## Notes and Observations

"""
            
            # Add notes based on metrics
            if metrics['exact_position_accuracy'] > 0.5:
                report_content += "- The model showed good accuracy in predicting exact positions.\n"
            else:
                report_content += "- The model had difficulty predicting exact positions.\n"
                
            if metrics['winner_prediction_accuracy'] > 0:
                report_content += "- The race winner was correctly predicted.\n"
            else:
                report_content += "- The race winner was not correctly predicted.\n"
                
            if metrics['podium_prediction_accuracy'] > 0.66:
                report_content += "- Podium prediction was highly accurate.\n"
            elif metrics['podium_prediction_accuracy'] > 0.33:
                report_content += "- Some podium positions were correctly predicted.\n"
            else:
                report_content += "- The model struggled to predict podium positions correctly.\n"
            
            # Write report to file
            with open(report_path, 'w') as f:
                f.write(report_content)
                
            self.agent_logger.info(f"Generated report at {report_path}")
            
            return report_path
            
        except Exception as e:
            self.agent_logger.error(f"Error generating report: {str(e)}")
            return None
    
    def _compile_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile an overall summary of prediction performance.
        
        Args:
            results: Dictionary with analysis results
            
        Returns:
            Dictionary with summary information
        """
        summary = {
            'race_info': results.get('race_info', {}),
            'metrics_summary': {},
            'best_prediction': None,
            'metrics_comparison': {}
        }
        
        # Extract metrics for each prediction type
        metrics_by_type = {}
        for pred_type, metrics in results.get('metrics', {}).items():
            metrics_by_type[pred_type] = metrics
        
        # Determine best prediction type based on multiple metrics
        if metrics_by_type:
            # Key metrics to consider
            key_metrics = ['rmse', 'exact_position_accuracy', 'winner_prediction_accuracy', 'podium_prediction_accuracy']
            
            # Scores for each prediction type
            scores = {pred_type: 0 for pred_type in metrics_by_type.keys()}
            
            # Compare each metric
            for metric in key_metrics:
                # For rmse, lower is better; for others, higher is better
                if metric == 'rmse':
                    best_type = min(metrics_by_type.items(), key=lambda x: x[1].get(metric, float('inf')))[0]
                else:
                    best_type = max(metrics_by_type.items(), key=lambda x: x[1].get(metric, 0))[0]
                
                # Add score to the best type
                scores[best_type] += 1
            
            # Determine overall best type
            best_type = max(scores.items(), key=lambda x: x[1])[0]
            summary['best_prediction'] = best_type
            
            # Add comparison of key metrics
            for metric in key_metrics:
                summary['metrics_comparison'][metric] = {
                    pred_type: metrics.get(metric) for pred_type, metrics in metrics_by_type.items()
                }
            
            # Add overall metrics summary
            for pred_type, metrics in metrics_by_type.items():
                summary['metrics_summary'][pred_type] = {
                    'rmse': metrics.get('rmse'),
                    'exact_accuracy': metrics.get('exact_position_accuracy'),
                    'winner_accuracy': metrics.get('winner_prediction_accuracy'),
                    'podium_accuracy': metrics.get('podium_prediction_accuracy')
                }
        
        return summary