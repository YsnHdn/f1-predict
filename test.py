import os
from pathlib import Path
from agents.supervisor import SupervisorAgent

# Créer le répertoire cache pour FastF1
cache_dir = Path('.fastf1_cache')
if not cache_dir.exists():
    os.makedirs(cache_dir)
    print(f"Created FastF1 cache directory: {cache_dir}")

# Initialiser le superviseur
supervisor = SupervisorAgent()

# Exécuter le workflow sans spécifier la course
context = {
    'year': 2025,  # année en cours
    'prediction_types': ['initial'],  # type de prédiction souhaité
}

# Exécuter le workflow complet
results = supervisor.execute(context)

# Afficher les résultats
print(f"Prochaine course: {results['race_info']['name']}")
if 'predictions' in results and 'initial' in results['predictions']:
    print("\nPrédictions:")
    predictions = results['predictions']['initial']['predictions']
    for i, (_, row) in enumerate(predictions.head(10).iterrows(), 1):
        print(f"{i}. {row['Driver']}")