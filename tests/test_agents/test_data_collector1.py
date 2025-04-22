import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import fastf1
import sys

# Ajoutez le répertoire parent au chemin Python pour pouvoir importer vos modules
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

# Créer le répertoire cache pour FastF1
cache_dir = Path('.fastf1_cache')
if not cache_dir.exists():
    os.makedirs(cache_dir)
    print(f"Created FastF1 cache directory: {cache_dir}")

# Importer les composants nécessaires
from agents.data_collector import DataCollectorAgent
from preprocessing.feature_engineering import F1FeatureEngineer
from models.initial_model import F1InitialModel

# Initialiser l'agent
collector = DataCollectorAgent()

# Récupérer le prochain Grand Prix
current_year = datetime.now().year
schedule = fastf1.get_event_schedule(current_year)
future_races = schedule[schedule['EventDate'] > datetime.now()]
if future_races.empty:
    raise ValueError(f"Aucune course future trouvée pour l'année {current_year}")
next_race = future_races.iloc[0]
race_name = next_race['EventName']
circuit_name = next_race['OfficialEventName']
race_date = next_race['EventDate']

print(f"Prochain GP: {race_name} à {circuit_name} le {race_date}")

# Configurer le contexte pour la collecte de données
context = {
    'year': current_year,
    'gp_name': race_name,
    'session_types': ['R', 'Q', 'FP1', 'FP2', 'FP3'],
    'historical_years': 3
}

# Exécuter la collecte
print("\n--- COLLECTE DES DONNÉES ---")
data_results = collector.run(context)

# Afficher les résultats
print("\n--- RÉSUMÉ DES DONNÉES COLLECTÉES ---")
print(f"Données collectées: {len(data_results.get('data_paths', {}))} fichiers")

for key, path in data_results.get('data_paths', {}).items():
    if os.path.exists(path):
        print(f"- {key}: {path}")
        if path.endswith('.csv'):
            try:
                df = pd.read_csv(path)
                print(f"  Dimensions: {df.shape}")
                print(f"  Colonnes: {', '.join(df.columns[:min(5, len(df.columns))])}...")
            except Exception as e:
                print(f"  Erreur lors de la lecture: {str(e)}")

# Analyser les données collectées
print("\n--- ANALYSE DES DONNÉES POUR FEATURES ---")
print("Recherche d'informations sur les pilotes actuels...")

# Rechercher les classements des pilotes
driver_standings_path = None
for key, path in data_results.get('data_paths', {}).items():
    if 'driver_standings' in key and os.path.exists(path):
        driver_standings_path = path
        break

if driver_standings_path:
    try:
        driver_standings = pd.read_csv(driver_standings_path)
        print(f"Classement des pilotes trouvé: {driver_standings.shape[0]} pilotes")
        print("Top 5 pilotes:")
        print(driver_standings[['Driver', 'Team', 'Points']].head(5))
    except Exception as e:
        print(f"Erreur lors de la lecture du classement des pilotes: {str(e)}")
else:
    print("Aucun classement des pilotes trouvé dans les données collectées")

# Rechercher les données historiques
historical_data_path = None
for key, path in data_results.get('data_paths', {}).items():
    if 'historical_combined' in key and os.path.exists(path):
        historical_data_path = path
        break

if historical_data_path:
    try:
        historical_data = pd.read_csv(historical_data_path)
        print(f"Données historiques trouvées: {historical_data.shape[0]} entrées")
        print(f"Courses précédentes à {race_name}:")
        years = historical_data['Year'].unique()
        print(f"Années disponibles: {', '.join(map(str, years))}")
    except Exception as e:
        print(f"Erreur lors de la lecture des données historiques: {str(e)}")
else:
    print("Aucune donnée historique combinée trouvée")

# Initialiser le module de feature engineering pour voir quelles features peuvent être créées
print("\n--- TEST DE GÉNÉRATION DE FEATURES ---")
feature_engineer = F1FeatureEngineer(scale_features=False)

# Si nous avons les pilotes actuels et des données historiques, créer un template pour le prochain GP
if driver_standings_path and historical_data_path:
    try:
        # Charger les données
        driver_standings = pd.read_csv(driver_standings_path)
        historical_data = pd.read_csv(historical_data_path)
        
        # Créer un template pour la prochaine course
        next_race_template = pd.DataFrame({
            'Driver': driver_standings['Driver'].values,
            'Team': driver_standings['Team'].values,
            'TrackName': [next_race['EventName']] * len(driver_standings),
            'GridPosition': [None] * len(driver_standings),  # Inconnu avant les qualifications
            'Year': [current_year] * len(driver_standings),
            'GrandPrix': [race_name] * len(driver_standings),
            'Date': [race_date] * len(driver_standings)
        })
        
        print(f"Template créé pour la prochaine course avec {len(next_race_template)} pilotes")
        
        # Essayer de générer les features
        features_df = feature_engineer.create_all_features(
            next_race_template, 
            historical_df=historical_data,
            encode_categorical=False
        )
        
        # Vérifier quelles features ont été créées
        print(f"Features générées: {features_df.shape[1]} colonnes")
        
        # Vérifier les features requises pour le modèle initial
        required_features = [
            'driver_avg_points', 'driver_avg_positions_gained', 'driver_finish_rate', 
            'driver_form_trend', 'driver_last3_avg_pos', 'team_avg_points', 
            'team_finish_rate', 'circuit_high_speed', 'circuit_street', 
            'circuit_technical', 'circuit_safety_car_rate', 'overtaking_difficulty'
        ]
        
        present_features = [f for f in required_features if f in features_df.columns]
        missing_features = [f for f in required_features if f not in features_df.columns]
        
        print(f"Features requises présentes: {len(present_features)}/{len(required_features)}")
        print(f"Features manquantes: {', '.join(missing_features)}")
        
        # Tester si les données peuvent être utilisées par le modèle initial
        initial_model = F1InitialModel()
        model_input = initial_model.prepare_features(features_df)
        
        print(f"Données prêtes pour le modèle initial: {model_input.shape[1]} features utilisables")
        
    except Exception as e:
        print(f"Erreur lors de la génération des features: {str(e)}")
else:
    print("Données insuffisantes pour générer des features")

print("\n--- FIN DU TEST ---")