"""
Script de test pour le client météo VisualCrossing et le WeatherMonitorAgent.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Importer les composants
from api.visualcrossing_client import VisualCrossingClient
from agents.weather_monitor import WeatherMonitorAgent

def test_visualcrossing_client():
    """Teste le client VisualCrossing."""
    print("\n=== Test du client VisualCrossing ===")
    
    # Créer le client
    client = VisualCrossingClient()
    
    # Tester les coordonnées des circuits
    circuit = "spa"
    coordinates = client.get_circuit_coordinates(circuit)
    print(f"Coordonnées pour {circuit}: {coordinates}")
    
    # Tester la météo actuelle
    current_weather = client.get_current_weather(circuit)
    print("\nMétéo actuelle:")
    print(current_weather.head())
    
    # Tester les prévisions
    forecast = client.get_weather_forecast(circuit, days=3)
    print("\nPrévisions (3 jours):")
    print(forecast.head())
    
    # Tester les alertes
    alerts = client.get_weather_alerts(circuit)
    print("\nAlertes météo:")
    print(alerts)
    
    # Tester l'impact météo
    impact = client.get_weather_impact_probability(circuit)
    print("\nImpact météo:")
    print(impact)
    
    return True

def test_weather_monitor_agent():
    """Teste le WeatherMonitorAgent avec le client VisualCrossing."""
    print("\n=== Test du WeatherMonitorAgent ===")
    
    # Créer un répertoire temporaire pour les tests
    test_dir = "test_data/weather"
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialiser l'agent
    agent = WeatherMonitorAgent(data_dir=test_dir)
    
    # Créer un contexte de test
    context = {
        'circuit': 'spa',
        'race_date': datetime.now().strftime('%Y-%m-%d'),
        'days_range': 2
    }
    
    # Exécuter l'agent
    try:
        results = agent.execute(context)
        print("\nRésultats de l'agent:")
        print(f"Circuit: {results['circuit']}")
        print(f"Date de course: {results['race_date']}")
        print(f"Nombre de fichiers: {len(results['data_paths'])}")
        
        # Afficher les conditions de course
        if 'race_conditions' in results:
            print("\nConditions de course prévues:")
            for key, value in results['race_conditions'].items():
                print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"Erreur lors du test de l'agent: {str(e)}")
        return False

def main():
    """Fonction principale."""
    # Charger les variables d'environnement
    load_dotenv()
    
    # Vérifier la clé API
    api_key = os.environ.get('VISUALCROSSING_API_KEY')
    if not api_key:
        print("ATTENTION: Clé API VisualCrossing non trouvée dans l'environnement.")
        print("Créez un fichier .env avec VISUALCROSSING_API_KEY=votre_clé_api")
        print("Les tests utiliseront des données simulées.")
    
    # Tester le client
    client_success = test_visualcrossing_client()
    
    # Tester l'agent si le client fonctionne
    if client_success:
        agent_success = test_weather_monitor_agent()
        if agent_success:
            print("\n✅ Tous les tests ont réussi!")
        else:
            print("\n❌ Test de l'agent échoué")
    else:
        print("\n❌ Test du client échoué")

if __name__ == "__main__":
    main()