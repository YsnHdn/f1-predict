"""
Feature diagnostics tool for F1 prediction project.
This script checks if all required features are available before model training.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add project root to path to import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import required modules
from models.initial_model import F1InitialModel
from preprocessing.feature_engineering import F1FeatureEngineer
from preprocessing.data_cleaning import F1DataCleaner
from api.fastf1_client import FastF1Client
from api.visualcrossing_client import VisualCrossingClient

def check_features(data_path: str, circuit: str = None, output_file: str = None):
    """
    Check if all required features are present in the data.
    
    Args:
        data_path: Path to the data file (CSV)
        circuit: Optional circuit name for weather features
        output_file: Optional path to save the diagnostic report
    """
    print(f"=== F1 Feature Diagnostics Tool ===")
    print(f"Checking features in: {data_path}")
    
    # Load data
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV file.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
        
    print(f"\nData loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Initialize the model to get required features
    initial_model = F1InitialModel()
    required_features = initial_model.features
    
    print(f"\n1. CHECKING REQUIRED MODEL FEATURES ({len(required_features)} features)")
    print("=" * 50)
    
    # Check each required feature
    missing_features = []
    present_features = []
    feature_status = {}
    
    for feature in required_features:
        if feature in df.columns:
            present_features.append(feature)
            feature_status[feature] = {
                'status': 'PRESENT',
                'non_null': df[feature].notna().sum(),
                'null_percentage': df[feature].isna().mean() * 100,
                'dtype': str(df[feature].dtype)
            }
        else:
            missing_features.append(feature)
            feature_status[feature] = {
                'status': 'MISSING',
                'non_null': 0,
                'null_percentage': 100.0,
                'dtype': 'N/A'
            }
    
    # Print feature status
    for feature, stats in feature_status.items():
        status_str = f"[✓]" if stats['status'] == 'PRESENT' else "[✗]"
        if stats['status'] == 'PRESENT':
            print(f"{status_str} {feature} ({stats['dtype']}) - {stats['non_null']}/{len(df)} values ({100-stats['null_percentage']:.1f}% complete)")
        else:
            print(f"{status_str} {feature} - MISSING")
    
    print(f"\nSummary: {len(present_features)}/{len(required_features)} required features present ({len(missing_features)} missing)")
    
    # Check base data for feature engineering
    print(f"\n2. CHECKING BASE DATA FOR FEATURE ENGINEERING")
    print("=" * 50)
    
    base_columns = [
        'Driver', 'Team', 'TrackName', 'Circuit', 'GridPosition', 'Position', 
        'Points', 'Year', 'Date', 'Status'
    ]
    
    base_status = {}
    for col in base_columns:
        matching_cols = [c for c in df.columns if c.lower() == col.lower()]
        if matching_cols:
            col_name = matching_cols[0]
            base_status[col] = {
                'status': 'PRESENT',
                'column_name': col_name,
                'non_null': df[col_name].notna().sum(),
                'null_percentage': df[col_name].isna().mean() * 100,
                'dtype': str(df[col_name].dtype)
            }
        else:
            base_status[col] = {
                'status': 'MISSING',
                'column_name': None
            }
    
    # Print base data status
    for col, stats in base_status.items():
        status_str = f"[✓]" if stats['status'] == 'PRESENT' else "[✗]"
        if stats['status'] == 'PRESENT':
            print(f"{status_str} {col} (as {stats['column_name']}) - {stats['non_null']}/{len(df)} values ({100-stats['null_percentage']:.1f}% complete)")
        else:
            print(f"{status_str} {col} - MISSING")
    
    # Check weather features
    print(f"\n3. CHECKING WEATHER FEATURES")
    print("=" * 50)
    
    weather_columns = [
        'temp_celsius', 'rain_mm', 'wind_speed_ms', 'weather_is_dry', 
        'weather_is_any_wet', 'weather_is_very_wet', 'weather_temp_mild', 
        'weather_temp_hot', 'weather_high_wind', 'racing_condition'
    ]
    
    weather_present = [col for col in weather_columns if col in df.columns]
    
    if len(weather_present) == 0:
        print("No weather features found in data.")
        
        # Check if we should try to generate weather features
        if circuit:
            print(f"\nAttempting to retrieve weather data for circuit: {circuit}")
            try:
                # Initialize weather client
                weather_client = VisualCrossingClient()
                
                # Get current date for demo
                current_date = datetime.now()
                
                # Get current weather
                current_weather = weather_client.get_current_weather(circuit)
                
                if current_weather is not None and not current_weather.empty:
                    print(f"\nCurrent weather example:")
                    print(current_weather.head(1).to_dict('records')[0])
                    
                    # Check which weather features would be available
                    available_weather = [col for col in weather_columns if col in current_weather.columns]
                    missing_weather = [col for col in weather_columns if col not in current_weather.columns]
                    
                    print(f"\nFromVisualCrossing API, we could get {len(available_weather)}/{len(weather_columns)} weather features:")
                    for col in available_weather:
                        print(f"[✓] {col}")
                    
                    for col in missing_weather:
                        print(f"[✗] {col}")
            except Exception as e:
                print(f"Error retrieving weather data: {str(e)}")
    else:
        print(f"Found {len(weather_present)}/{len(weather_columns)} weather features")
        for col in weather_columns:
            status_str = f"[✓]" if col in df.columns else "[✗]"
            if col in df.columns:
                print(f"{status_str} {col} ({df[col].dtype}) - {df[col].notna().sum()}/{len(df)} values ({df[col].notna().mean()*100:.1f}% complete)")
            else:
                print(f"{status_str} {col} - MISSING")
    
    # Test feature engineering
    print(f"\n4. TESTING FEATURE ENGINEERING")
    print("=" * 50)
    
    # Initialize feature engineer
    feature_engineer = F1FeatureEngineer(scale_features=False)
    
    # Check which feature generation methods can be applied
    methods = [
        'create_grid_position_features',
        'create_qualifying_features',
        'create_team_performance_features',
        'create_driver_performance_features',
        'create_circuit_features',
        'create_weather_impact_features',
        'create_interaction_features'
    ]
    
    for method in methods:
        try:
            # Try to apply the method
            result_df = getattr(feature_engineer, method)(df)
            
            # Compare columns before and after
            new_columns = [col for col in result_df.columns if col not in df.columns]
            
            if new_columns:
                print(f"[✓] {method} - Added {len(new_columns)} new features:")
                for col in new_columns[:5]:  # Show first 5 new features
                    print(f"    - {col}")
                if len(new_columns) > 5:
                    print(f"    - ... and {len(new_columns)-5} more")
            else:
                print(f"[✗] {method} - No new features added")
                
        except Exception as e:
            print(f"[✗] {method} - Error: {str(e)}")
    
    # Try full feature generation
    print(f"\n5. ATTEMPTING FULL FEATURE GENERATION")
    print("=" * 50)
    
    try:
        # Clean data first
        cleaner = F1DataCleaner()
        cleaned_df = cleaner.clean_data_types(df)
        
        # Generate all features
        features_df = feature_engineer.create_all_features(cleaned_df, encode_categorical=False)
        
        # Check which required features are now available
        final_present_features = [feature for feature in required_features if feature in features_df.columns]
        final_missing_features = [feature for feature in required_features if feature not in features_df.columns]
        
        print(f"After feature engineering: {len(final_present_features)}/{len(required_features)} required features present")
        
        if final_missing_features:
            print(f"\nStill missing {len(final_missing_features)} required features:")
            for feature in final_missing_features:
                print(f"[✗] {feature}")
        else:
            print(f"\n[✓] All required features are now available!")
            
        # Extra diagnostics - check for NaN values in required features
        non_null_counts = features_df[final_present_features].notna().sum()
        completion_rates = non_null_counts / len(features_df) * 100
        
        print(f"\nFeature completion rates:")
        for feature, rate in zip(final_present_features, completion_rates):
            status = "GOOD" if rate >= 90 else "WARNING" if rate >= 50 else "CRITICAL"
            print(f"{feature}: {rate:.1f}% complete - {status}")
            
        # Overall readiness assessment
        if len(final_missing_features) == 0 and all(rate >= 50 for rate in completion_rates):
            print(f"\n[✓] READY FOR TRAINING - All required features present with acceptable completion rates")
        elif len(final_missing_features) <= 3 and all(rate >= 30 for rate in completion_rates):
            print(f"\n[!] PARTIALLY READY - Most features present but some issues need attention")
        else:
            print(f"\n[✗] NOT READY - Significant feature issues must be resolved before training")
            
    except Exception as e:
        print(f"Error during full feature generation: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # Save diagnostic report if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write("=== F1 FEATURE DIAGNOSTICS REPORT ===\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data file: {data_path}\n")
                f.write(f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n")
                
                f.write("1. REQUIRED FEATURES STATUS\n")
                f.write("=========================\n")
                for feature, stats in feature_status.items():
                    status_str = "PRESENT" if stats['status'] == 'PRESENT' else "MISSING"
                    f.write(f"{feature}: {status_str}\n")
                
                f.write(f"\nSummary: {len(present_features)}/{len(required_features)} features present\n\n")
                
                if 'final_present_features' in locals() and 'final_missing_features' in locals():
                    f.write("2. AFTER FEATURE ENGINEERING\n")
                    f.write("=========================\n")
                    f.write(f"Present: {len(final_present_features)}/{len(required_features)}\n")
                    if final_missing_features:
                        f.write(f"Missing: {', '.join(final_missing_features)}\n")
                
            print(f"\nDiagnostic report saved to {output_file}")
        except Exception as e:
            print(f"Error saving diagnostic report: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Check features for F1 prediction model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data file (CSV)")
    parser.add_argument("--circuit", type=str, help="Circuit name for weather feature retrieval")
    parser.add_argument("--output", type=str, help="Path to save diagnostic report")
    
    args = parser.parse_args()
    
    check_features(args.data_path, args.circuit, args.output)

if __name__ == "__main__":
    main()