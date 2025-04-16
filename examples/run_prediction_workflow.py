"""
Example script to demonstrate the F1 prediction workflow using the agent system.
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supervisor import SupervisorAgent
from agents.utils.logging import setup_logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run F1 prediction workflow')
    
    parser.add_argument('--race-name', type=str, required=True,
                      help='Name of the race (e.g., "Belgian Grand Prix")')
    
    parser.add_argument('--circuit', type=str, required=True,
                      help='Circuit identifier (e.g., "spa")')
    
    parser.add_argument('--race-date', type=str, required=True,
                      help='Date of the race in YYYY-MM-DD format')
    
    parser.add_argument('--prediction-types', type=str, nargs='+',
                      default=['initial', 'pre_race', 'race_day'],
                      help='Types of predictions to generate')
    
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to store output files')
    
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      default='INFO', help='Logging level')
    
    return parser.parse_args()

def main():
    """Run the F1 prediction workflow."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger('workflow', level=log_level)
    
    logger.info(f"Starting F1 prediction workflow for {args.race_name}")
    
    try:
        # Initialize the supervisor agent
        supervisor = SupervisorAgent(output_dir=args.output_dir)
        
        # Prepare context for execution
        context = {
            'race_name': args.race_name,
            'circuit': args.circuit,
            'race_date': args.race_date,
            'prediction_types': args.prediction_types
        }
        
        # Execute the workflow
        results = supervisor.execute(context)
        
        # Print summary
        logger.info("Workflow completed successfully")
        logger.info(f"Race: {args.race_name}")
        logger.info(f"Circuit: {args.circuit}")
        logger.info(f"Date: {args.race_date}")
        
        # Print prediction results if available
        for pred_type, pred_results in results.get('predictions', {}).items():
            if pred_results and 'predictions' in pred_results:
                predictions = pred_results['predictions']
                logger.info(f"\n{pred_type.upper()} PREDICTION:")
                
                # Display top 5 predictions
                top5 = predictions.head(5)
                for i, (_, row) in enumerate(top5.iterrows(), 1):
                    logger.info(f"  {i}. {row['Driver']}")
        
        logger.info(f"Detailed results saved to {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())