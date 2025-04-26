def clean_data_types(df):
    """
    Nettoie et corrige les types de données pour éviter les conflits lors de l'entraînement.
    
    Args:
        df: DataFrame à nettoyer
        
    Returns:
        DataFrame avec des types de données cohérents
    """
    import pandas as pd
    result_df = df.copy()
    
    # Colonnes à convertir en numérique
    numeric_cols = ['Position', 'GridPosition', 'Points', 'driver_avg_points', 
                   'driver_avg_positions_gained', 'driver_finish_rate', 
                   'driver_form_trend', 'driver_last3_avg_pos', 'team_avg_points', 
                   'team_finish_rate', 'circuit_safety_car_rate', 'overtaking_difficulty']
    
    for col in numeric_cols:
        if col in result_df.columns:
            # Convertir en numérique, les non-convertibles deviennent NaN
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    # Colonnes booléennes ou indicateurs (0/1)
    bool_cols = ['circuit_high_speed', 'circuit_street', 'circuit_technical',
                'weather_is_dry', 'weather_is_any_wet', 'weather_temp_mild',
                'weather_temp_hot', 'weather_high_wind']
    
    for col in bool_cols:
        if col in result_df.columns:
            # Convertir en entiers (0 ou 1)
            if result_df[col].dtype != 'int64':
                result_df[col] = result_df[col].astype(int)
    
    # Traiter séparément les colonnes de date
    date_cols = ['Date']
    for col in date_cols:
        if col in result_df.columns:
            # Pour l'entraînement du modèle, supprimer les colonnes de date 
            # car elles causent des conflits de types
            result_df = result_df.drop(col, axis=1)
    
    # S'assurer que tous les NaN restants sont remplacés par des valeurs par défaut
    result_df = result_df.fillna({
        'Position': result_df['Position'].mean() if 'Position' in result_df.columns else 10,
        'GridPosition': result_df['GridPosition'].mean() if 'GridPosition' in result_df.columns else 10,
        'Points': 0,
        'driver_avg_points': 0,
        'driver_avg_positions_gained': 0,
        'driver_finish_rate': 0.9,
        'driver_form_trend': 0,
        'driver_last3_avg_pos': 10,
        'team_avg_points': 0,
        'team_finish_rate': 0.9,
        'circuit_safety_car_rate': 0.2,
        'overtaking_difficulty': 0.5
    })
    
    # Vérification finale pour s'assurer qu'il n'y a aucune valeur NaN restante
    for col in result_df.columns:
        if result_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].fillna(0)
            else:
                result_df[col] = result_df[col].fillna('unknown')
    
    return result_df