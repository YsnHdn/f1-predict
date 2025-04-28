"""
Test script for the modified WeatherMonitorAgent.
This script tests the functionality of getting historical weather data from FastF1.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the modified WeatherMonitorAgent
from agents.weather_monitor import WeatherMonitorAgent

def main():
    """Main function to test the modified WeatherMonitorAgent."""
    print("=== Testing Modified WeatherMonitorAgent ===")
    
    # Create FastF1 cache directory if it doesn't exist
    cache_dir = Path('.fastf1_cache')
    if not cache_dir.exists():
        os.makedirs(cache_dir)
        print(f"Created FastF1 cache directory: {cache_dir}")
    
    # Initialize the WeatherMonitorAgent
    agent = WeatherMonitorAgent()
    print("Initialized WeatherMonitorAgent")
    
    # Test circuit and dates
    circuit = "monza"  # Change to any circuit you want to test
    race_date = datetime.now() + timedelta(days=30)  # Future race date
    days_range = 3
    
    print(f"Testing historical weather data for {circuit}")
    
    # Calculate date range
    start_date = race_date - timedelta(days=days_range)
    end_date = race_date + timedelta(days=days_range)
    
    # Test getting historical weather data from FastF1
    historical_data = agent.get_historical_weather_from_fastf1(
        circuit, start_date, end_date, years_back=3
    )
    
    # Check results
    if historical_data:
        print(f"Successfully retrieved {len(historical_data)} historical weather datasets")
        
        # Print sample of the first dataset
        if historical_data[0] is not None and not historical_data[0].empty:
            print("\nSample of historical weather data:")
            print(historical_data[0].head())
            
            # Get columns to verify
            print("\nColumns in historical weather data:")
            print(historical_data[0].columns.tolist())
            
            # Check for required weather features
            required_cols = [
                'temp_celsius', 'rain_mm', 'wind_speed_ms', 
                'weather_is_dry', 'weather_is_any_wet', 'weather_temp_mild'
            ]
            
            missing_cols = [col for col in required_cols if col not in historical_data[0].columns]
            
            if missing_cols:
                print(f"\nWARNING: Missing required weather columns: {missing_cols}")
            else:
                print("\nAll required weather features are present in the data")
                
        else:
            print("Retrieved historical data is empty")
    else:
        print("Failed to retrieve historical weather data from FastF1")
    
    # Test the full execute method with a context
    print("\n=== Testing full execute method ===")
    context = {
        'circuit': circuit,
        'race_date': race_date,
        'days_range': days_range,
    }
    
    try:
        results = agent.execute(context)
        
        # Check if historical data was retrieved and saved
        if 'historical_data' in results and results['historical_data']:
            print(f"Successfully saved historical weather data to: {results['historical_data']}")
        else:
            print("No historical weather data was saved")
            
        # Check if forecast data was retrieved and saved
        if 'forecast' in results and results['forecast']:
            print(f"Successfully saved forecast weather data to: {results['forecast']}")
        else:
            print("No forecast weather data was saved")
            
        # Check if race conditions were extracted
        if 'race_conditions' in results:
            print("\nExtracted race conditions:")
            for key, value in results['race_conditions'].items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error executing weather monitor: {str(e)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()