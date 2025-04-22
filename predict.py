import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

# Ajouter le répertoire racine au path Python
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importer les composants nécessaires
from agents.data_collector import DataCollectorAgent
from agents.weather_monitor import WeatherMonitorAgent
from preprocessing.feature_engineering import F1FeatureEngineer
from models.initial_model import F1InitialModel
from preprocessing.data_cleaning import F1DataCleaner

# Configurer la mise en page des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

def setup_directories():
    """Créer les répertoires nécessaires s'ils n'existent pas."""
    directories = ['data/raw', 'data/processed', 'models', 'output', 'predictions']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Créer le répertoire cache pour FastF1
    cache_dir = Path('.fastf1_cache')
    if not cache_dir.exists():
        os.makedirs(cache_dir)
        print(f"Created FastF1 cache directory: {cache_dir}")

def get_next_race():
    """Récupérer les informations sur la prochaine course."""
    # Initialiser un DataCollectorAgent temporaire pour récupérer le calendrier
    temp_collector = DataCollectorAgent()
    
    # Récupérer la prochaine course
    current_year = datetime.now().year
    context = {'year': current_year}
    
    results = temp_collector.execute(context)
    
    return results['next_race']

def collect_data(next_race):
    """Collecter toutes les données nécessaires pour la prédiction."""
    print("\n--- COLLECTE DES DONNÉES ---")
    
    # Initialiser le DataCollectorAgent
    data_collector = DataCollectorAgent()
    
    # Collecter les données de courses et historiques
    race_context = {
        'year': datetime.now().year,
        'gp_name': next_race['name'],
        'historical_years': 5  # Utiliser 5 années pour plus de données
    }
    
    data_results = data_collector.run(race_context)
    
    # Extraire les chemins des fichiers importants
    data_paths = data_results.get('data_paths', {})
    
    print(f"Données collectées pour {next_race['name']} ({len(data_paths)} fichiers)")
    
    return data_results

def get_weather_data(next_race):
    """Récupérer les données météorologiques pour la prochaine course."""
    print("\n--- COLLECTE DES DONNÉES MÉTÉO ---")
    
    # Initialiser le WeatherMonitorAgent
    weather_monitor = WeatherMonitorAgent()
    
    # Configurer le contexte pour la collecte météo
    weather_context = {
        'circuit': next_race['circuit'],
        'race_date': next_race['date'],
        'days_range': 3
    }
    
    # Récupérer les données météo
    weather_results = weather_monitor.run(weather_context)
    
    # Extraire les conditions météo pour la course
    race_conditions = weather_results.get('race_conditions', {})
    
    print(f"Conditions météo récupérées pour {next_race['name']}")
    
    return weather_results

def prepare_data_for_prediction(data_results, weather_results):
    """Préparer les données pour la prédiction."""
    print("\n--- PRÉPARATION DES DONNÉES ---")
    
    # Initialiser le nettoyeur de données
    cleaner = F1DataCleaner()
    
    # Charger les données
    data_paths = data_results.get('data_paths', {})
    
    # Charger le classement actuel des pilotes
    driver_standings = None
    if 'driver_standings' in data_paths:
        driver_standings = pd.read_csv(data_paths['driver_standings'])
    
    # Charger les informations sur les pilotes actuels
    current_drivers = None
    if 'current_drivers' in data_paths:
        current_drivers = pd.read_csv(data_paths['current_drivers'])
    
    # Charger les données historiques
    historical_data = None
    for key, path in data_paths.items():
        if 'historical_combined' in key and os.path.exists(path):
            historical_data = pd.read_csv(path)
            break
    
    # Créer un template pour la prochaine course
    if driver_standings is not None and current_drivers is not None:
        # Fusionner les données des pilotes avec les standings
        driver_info = pd.merge(
            driver_standings, 
            current_drivers[['Abbreviation', 'TeamName', 'DriverNumber']],
            left_on='Driver',
            right_on='Abbreviation',
            how='left'
        )
        
        # Créer un template pour la prochaine course
        next_race_info = data_results['next_race']
        next_race_template = pd.DataFrame({
            'Driver': driver_info['Driver'].values,
            'Team': driver_info['Team'].values,
            'TrackName': [next_race_info['circuit']] * len(driver_info),
            'Year': [datetime.now().year] * len(driver_info),
            'GrandPrix': [next_race_info['name']] * len(driver_info),
            'Date': [next_race_info['date']] * len(driver_info)
        })
        
        # Nettoyer et standardiser les noms
        next_race_template = cleaner.standardize_driver_names(next_race_template)
        next_race_template = cleaner.standardize_team_names(next_race_template)
        next_race_template = cleaner.standardize_circuit_names(next_race_template, circuit_col='TrackName')
        
        # Nettoyer les données historiques
        if historical_data is not None:
            historical_data = cleaner.standardize_driver_names(historical_data)
            historical_data = cleaner.standardize_team_names(historical_data)
            if 'TrackName' in historical_data.columns:
                historical_data = cleaner.standardize_circuit_names(historical_data, circuit_col='TrackName')
        
        # Ajouter les données météorologiques si disponibles
        if weather_results and 'race_conditions' in weather_results:
            race_conditions = weather_results['race_conditions']
            
            # Ajouter chaque condition météo au template
            for key, value in race_conditions.items():
                next_race_template[f'weather_{key}'] = value
        
        print(f"Template créé pour la prochaine course avec {len(next_race_template)} pilotes")
        
        return {
            'next_race_template': next_race_template,
            'historical_data': historical_data,
            'driver_standings': driver_standings,
            'current_drivers': current_drivers
        }
    else:
        print("Données insuffisantes pour créer un template")
        return None

def generate_features(prepared_data):
    """Générer les features pour la prédiction."""
    print("\n--- GÉNÉRATION DES FEATURES ---")
    
    if prepared_data is None:
        print("Données préparées non disponibles")
        return None
    
    # Initialiser le module de feature engineering
    feature_engineer = F1FeatureEngineer(scale_features=False)
    
    # Générer les features
    next_race_template = prepared_data['next_race_template']
    historical_data = prepared_data['historical_data']
    
    features_df = feature_engineer.create_all_features(
        next_race_template, 
        historical_df=historical_data,
        encode_categorical=False
    )
    
    # Vérifier les features requises
    required_features = [
        'driver_avg_points', 'driver_avg_positions_gained', 'driver_finish_rate', 
        'driver_form_trend', 'driver_last3_avg_pos', 'team_avg_points', 
        'team_finish_rate', 'circuit_high_speed', 'circuit_street', 
        'circuit_technical', 'circuit_safety_car_rate', 'overtaking_difficulty'
    ]
    
    present_features = [f for f in required_features if f in features_df.columns]
    missing_features = [f for f in required_features if f not in features_df.columns]
    
    print(f"Features requises présentes: {len(present_features)}/{len(required_features)}")
    if missing_features:
        print(f"Features manquantes: {', '.join(missing_features)}")
    
    return features_df

def train_model(features_df, historical_data):
    """Entraîner le modèle de prédiction initial."""
    print("\n--- ENTRAÎNEMENT DU MODÈLE ---")
    
    if features_df is None or historical_data is None:
        print("Données insuffisantes pour l'entraînement")
        return None
    
    # Créer une version nettoyée des données historiques pour l'entraînement
    train_data = historical_data.copy()
    
    # S'assurer que Position et Points sont numériques
    if 'Position' in train_data.columns:
        train_data['Position'] = pd.to_numeric(train_data['Position'], errors='coerce')
    
    if 'Points' in train_data.columns:
        train_data['Points'] = pd.to_numeric(train_data['Points'], errors='coerce')
    
    # Initialiser le modèle
    initial_model = F1InitialModel(estimator='rf', target='Position')
    
    # Générer les features pour les données d'entraînement
    feature_engineer = F1FeatureEngineer(scale_features=False)
    train_features = feature_engineer.create_all_features(train_data, encode_categorical=False)
    
    # Filtrer pour les lignes avec une Position valide
    valid_rows = train_features['Position'].notna()
    X_train = train_features[valid_rows]
    y_train = X_train['Position']
    
    # Entraîner le modèle
    initial_model.train(X_train, y_train)
    
    # Évaluer la performance sur les données d'entraînement
    metrics = initial_model.evaluate(X_train, y_train)
    
    print("Métriques d'entraînement:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return initial_model

def make_predictions(model, features_df):
    """Faire des prédictions pour la prochaine course."""
    print("\n--- GÉNÉRATION DES PRÉDICTIONS ---")
    
    if model is None or features_df is None:
        print("Modèle ou features non disponibles")
        return None
    
    # Faire des prédictions
    predictions = model.predict_race_results(features_df)
    
    # Afficher les prédictions
    print("Prédictions pour la prochaine course:")
    for i, (_, row) in enumerate(predictions.head(10).iterrows(), 1):
        print(f"{i}. {row['Driver']}")
    
    return predictions

def visualize_predictions(predictions, next_race_info):
    """Visualiser les prédictions."""
    print("\n--- VISUALISATION DES PRÉDICTIONS ---")
    
    if predictions is None:
        print("Prédictions non disponibles")
        return
    
    # Créer un dossier pour les visualisations
    os.makedirs('output/visualizations', exist_ok=True)
    
    # Créer un graphique pour les prédictions
    plt.figure(figsize=(12, 8))
    
    # Utiliser seulement les 10 premiers pilotes pour la lisibilité
    top_predictions = predictions.head(10).copy()
    
    # Inverser l'ordre pour que le 1er soit en haut
    top_predictions = top_predictions.sort_values('PredictedPosition', ascending=False)
    
    # Créer un graphique à barres horizontales
    bars = plt.barh(top_predictions['Driver'], 
                   top_predictions['PredictedPosition'].max() - top_predictions['PredictedPosition'] + 1,
                   color='skyblue')
    
    # Ajouter les étiquettes
    plt.xlabel('Position prédite')
    plt.ylabel('Pilote')
    plt.title(f'Prédiction pour {next_race_info["name"]} {datetime.now().year}')
    
    # Ajouter des étiquettes à l'intérieur des barres
    for i, bar in enumerate(bars):
        position = int(top_predictions['PredictedPosition'].iloc[i])
        plt.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
                str(position), ha='center', va='center', fontweight='bold')
    
    # Inverser l'axe y pour que le meilleur pilote soit en haut
    plt.gca().invert_yaxis()
    
    # Ajouter une légende en bas
    plt.figtext(0.5, 0.01, f"Prédiction générée le {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
               ha='center', fontsize=10)
    
    # Enregistrer le graphique
    filename = f"output/visualizations/{datetime.now().strftime('%Y%m%d')}_{next_race_info['name'].replace(' ', '_')}_prediction.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualisation enregistrée: {filename}")

def save_predictions(predictions, next_race_info):
    """Enregistrer les prédictions dans un fichier."""
    print("\n--- SAUVEGARDE DES PRÉDICTIONS ---")
    
    if predictions is None:
        print("Prédictions non disponibles")
        return
    
    # Créer un dossier pour les prédictions
    os.makedirs('predictions', exist_ok=True)
    
    # Créer un dictionnaire avec les informations de prédiction
    prediction_data = {
        'prediction_type': 'initial',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'race_info': {
            'name': next_race_info['name'],
            'circuit': next_race_info['circuit'],
            'date': next_race_info['date']
        },
        'predictions': predictions.to_dict(orient='records')
    }
    
    # Enregistrer au format JSON
    filename = f"predictions/{datetime.now().strftime('%Y%m%d')}_{next_race_info['name'].replace(' ', '_')}_initial.json"
    
    import json
    with open(filename, 'w') as f:
        json.dump(prediction_data, f, indent=2, default=str)
    
    print(f"Prédictions enregistrées: {filename}")

def main():
    """Fonction principale qui orchestre l'ensemble du processus de prédiction."""
    print("=== SYSTÈME DE PRÉDICTION F1 ===")
    
    # Créer les répertoires nécessaires
    setup_directories()
    
    # Récupérer les informations sur la prochaine course
    next_race = get_next_race()
    print(f"Prochaine course: {next_race['name']} à {next_race['circuit']} le {next_race['date']}")
    
    # Collecter les données
    data_results = collect_data(next_race)
    
    # Récupérer les données météo
    weather_results = get_weather_data(next_race)
    
    # Préparer les données pour la prédiction
    prepared_data = prepare_data_for_prediction(data_results, weather_results)
    
    # Générer les features
    features_df = generate_features(prepared_data)
    
    # Entraîner le modèle
    model = train_model(features_df, prepared_data['historical_data'])
    
    # Faire des prédictions
    predictions = make_predictions(model, features_df)
    
    # Visualiser les prédictions
    visualize_predictions(predictions, next_race)
    
    # Enregistrer les prédictions
    save_predictions(predictions, next_race)
    
    print("\n=== PRÉDICTION TERMINÉE ===")

if __name__ == "__main__":
    main()