"""
Client pour l'API VisualCrossing Weather.
Ce module gère la récupération des données météorologiques pour les circuits de F1.
"""

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Configurer logging
logger = logging.getLogger(__name__)

class VisualCrossingClient:
    """
    Client pour l'API VisualCrossing Weather.
    Permet de récupérer les prévisions météo et les données historiques.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client VisualCrossing.
        
        Args:
            api_key: Clé API pour VisualCrossing (optionnel si définie dans .env)
        """
        # Charger les variables d'environnement
        load_dotenv()
        
        self.api_key = api_key or os.environ.get('VISUALCROSSING_API_KEY')
        
        if not self.api_key:
            logger.warning("Aucune clé API VisualCrossing trouvée. Utilisez une clé valide pour des données précises.")
        
        # Mapping des circuits F1 vers leurs coordonnées
        self.circuit_coordinates = {
            'spa': (50.4372, 5.9714),  # Spa-Francorchamps
            'monza': (45.6156, 9.2811),  # Monza
            'silverstone': (52.0706, -1.0174),  # Silverstone
            'monaco': (43.7347, 7.4205),  # Monaco
            'bahrain': (26.0325, 50.5106),  # Bahrain
            'jeddah': (21.6319, 39.1044),  # Jeddah
            'albert_park': (-37.8497, 144.9680),  # Melbourne
            'imola': (44.3439, 11.7167),  # Imola
            'miami': (25.9581, -80.2392),  # Miami
            'catalunya': (41.5689, 2.2611),  # Barcelona
            'baku': (40.3725, 49.8533),  # Baku
            'hungaroring': (47.5830, 19.2526),  # Hungaroring
            'zandvoort': (52.3888, 4.5411),  # Zandvoort
            'suzuka': (34.8431, 136.5410),  # Suzuka
            'singapore': (1.2914, 103.8550),  # Singapore
            'americas': (30.1345, -97.6358),  # Circuit of the Americas
            'mexico': (19.4042, -99.0907),  # Mexico City
            'interlagos': (-23.7036, -46.6997),  # Interlagos
            'las_vegas': (36.2722, -115.0077),  # Las Vegas
            'yas_marina': (24.4672, 54.6031),  # Yas Marina
            'losail': (25.4882, 51.4536),  # Losail
            'red_bull_ring': (47.2225, 14.7607),  # Red Bull Ring
            'paul_ricard': (43.2506, 5.7931),  # Paul Ricard
            'rodriguez': (19.4042, -99.0907),  # Rodriguez (Mexico)
            'sochi': (43.4057, 39.9578),  # Sochi
            'istanbul': (40.9517, 29.4050),  # Istanbul
            'portimao': (37.2306, -8.6267),  # Portimao
            'nurburgring': (50.3356, 6.9475),  # Nurburgring
            'mugello': (43.9975, 11.3719),  # Mugello
            'hockenheim': (49.3278, 8.5694),  # Hockenheim
        }
    
    def get_circuit_coordinates(self, circuit: str) -> Optional[tuple]:
        """
        Récupère les coordonnées d'un circuit F1.
        
        Args:
            circuit: Nom ou identifiant du circuit
            
        Returns:
            Tuple (latitude, longitude) ou None si non trouvé
        """
        # Normaliser le nom du circuit
        circuit_lower = circuit.lower().replace(' ', '_')
        
        # Essayer de trouver le circuit exact
        if circuit_lower in self.circuit_coordinates:
            return self.circuit_coordinates[circuit_lower]
        
        # Essayer de trouver un circuit qui contient le terme
        for key, coords in self.circuit_coordinates.items():
            if circuit_lower in key or key in circuit_lower:
                logger.info(f"Circuit correspondant trouvé: {key} pour '{circuit}'")
                return coords
        
        # Si le circuit n'est pas trouvé, essayer de déterminer le pays
        country_mapping = {
            'belgium': 'spa',
            'italy': 'monza',
            'uk': 'silverstone',
            'monaco': 'monaco',
            'bahrain': 'bahrain',
            'saudi': 'jeddah',
            'australia': 'albert_park',
            'imola': 'imola',
            'miami': 'miami',
            'spain': 'catalunya',
            'azerbaijan': 'baku',
            'hungary': 'hungaroring',
            'netherlands': 'zandvoort',
            'japan': 'suzuka',
            'singapore': 'singapore',
            'usa': 'americas',
            'mexico': 'rodriguez',
            'brazil': 'interlagos',
            'vegas': 'las_vegas',
            'abu dhabi': 'yas_marina',
            'qatar': 'losail',
            'austria': 'red_bull_ring',
            'france': 'paul_ricard',
            'russia': 'sochi',
            'turkey': 'istanbul',
            'portugal': 'portimao',
            'germany': 'hockenheim',
        }
        
        for country, circuit_key in country_mapping.items():
            if country in circuit_lower:
                logger.info(f"Circuit par pays trouvé: {circuit_key} pour '{circuit}'")
                return self.circuit_coordinates.get(circuit_key)
        
        logger.warning(f"Aucune coordonnée trouvée pour le circuit: {circuit}")
        return None
    
    def get_simulated_weather(self, circuit: str, date) -> Dict[str, Any]:
        """
        Génère des données météo simulées en fallback.
        
        Args:
            circuit: Nom du circuit
            date: Date pour la météo
            
        Returns:
            Dictionnaire avec les données météo simulées
        """
        import random
        
        # Déterminer la saison
        month = date.month if isinstance(date, datetime) else datetime.strptime(date, '%Y-%m-%d').month
        
        if 3 <= month <= 5:  # Printemps
            base_temp = 18
            rain_prob = 0.3
        elif 6 <= month <= 8:  # Été
            base_temp = 25
            rain_prob = 0.2
        elif 9 <= month <= 11:  # Automne
            base_temp = 16
            rain_prob = 0.4
        else:  # Hiver
            base_temp = 10
            rain_prob = 0.5
        
        # Ajuster selon le circuit
        circuit_lower = circuit.lower()
        if 'spa' in circuit_lower or 'silverstone' in circuit_lower:
            # Circuits connus pour la pluie
            rain_prob += 0.2
            base_temp -= 2
        elif 'bahrain' in circuit_lower or 'abu dhabi' in circuit_lower:
            # Circuits chauds et secs
            rain_prob -= 0.15
            base_temp += 5
        
        # Ajouter de l'aléatoire
        actual_temp = base_temp + random.uniform(-3, 3)
        actual_rain = random.random() < rain_prob
        actual_rain_mm = random.uniform(0, 10) if actual_rain else 0
        actual_wind = random.uniform(0, 12)
        
        # Déterminer la condition de course
        if actual_rain_mm > 5:
            racing_condition = 'very_wet'
        elif actual_rain_mm > 2:
            racing_condition = 'wet'
        elif actual_rain_mm > 0.5:
            racing_condition = 'damp'
        else:
            racing_condition = 'dry'
        
        return {
            'temp_celsius': actual_temp,
            'rain_mm': actual_rain_mm,
            'wind_speed_ms': actual_wind,
            'weather_is_dry': racing_condition == 'dry',
            'weather_is_any_wet': racing_condition != 'dry',
            'weather_is_very_wet': racing_condition == 'very_wet',
            'weather_temp_mild': 15 <= actual_temp <= 25,
            'weather_temp_hot': actual_temp > 25,
            'weather_high_wind': actual_wind > 8,
            'racing_condition': racing_condition
        }
    
    def get_current_weather(self, circuit: str) -> pd.DataFrame:
        """
        Récupère les conditions météo actuelles pour un circuit.
        
        Args:
            circuit: Nom ou identifiant du circuit
            
        Returns:
            DataFrame avec les données météo actuelles
        """
        # Récupérer les coordonnées du circuit
        coordinates = self.get_circuit_coordinates(circuit)
        
        if not coordinates or not self.api_key:
            # Si pas de coordonnées ou pas de clé API, utiliser la simulation
            simulated_data = self.get_simulated_weather(circuit, datetime.now())
            return pd.DataFrame([simulated_data])
        
        lat, lon = coordinates
        
        # URL de l'API pour la météo actuelle
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today?key={self.api_key}&unitGroup=metric&include=current"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            current = data.get('currentConditions', {})
            
            # Déterminer la condition de course
            precip = current.get('precip', 0)
            if precip > 5:
                racing_condition = 'very_wet'
            elif precip > 2:
                racing_condition = 'wet'
            elif precip > 0.5:
                racing_condition = 'damp'
            else:
                racing_condition = 'dry'
            
            weather_data = {
                'temp_celsius': current.get('temp'),
                'rain_mm': precip,
                'wind_speed_ms': current.get('windspeed') * 0.44704 if 'windspeed' in current else 0,  # mph à m/s
                'humidity': current.get('humidity'),
                'cloud_cover': current.get('cloudcover'),
                'conditions': current.get('conditions'),
                'timestamp': current.get('datetime'),
                'weather_is_dry': racing_condition == 'dry',
                'weather_is_any_wet': racing_condition != 'dry',
                'weather_is_very_wet': racing_condition == 'very_wet',
                'weather_temp_mild': 15 <= current.get('temp', 22) <= 25,
                'weather_temp_hot': current.get('temp', 22) > 25,
                'weather_high_wind': (current.get('windspeed', 0) * 0.44704) > 8,
                'racing_condition': racing_condition
            }
            
            return pd.DataFrame([weather_data])
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données météo actuelles: {str(e)}")
            # Utiliser des données simulées en cas d'erreur
            simulated_data = self.get_simulated_weather(circuit, datetime.now())
            return pd.DataFrame([simulated_data])
    
    def get_weather_forecast(self, circuit: str, days: int = 7) -> pd.DataFrame:
        """
        Récupère les prévisions météo pour un circuit.
        
        Args:
            circuit: Nom ou identifiant du circuit
            days: Nombre de jours de prévision à récupérer
            
        Returns:
            DataFrame avec les prévisions météo
        """
        # Récupérer les coordonnées du circuit
        coordinates = self.get_circuit_coordinates(circuit)
        
        if not coordinates or not self.api_key:
            # Si pas de coordonnées ou pas de clé API, utiliser la simulation
            forecast_data = []
            start_date = datetime.now()
            for i in range(days):
                forecast_date = start_date + timedelta(days=i)
                daily_forecast = self.get_simulated_weather(circuit, forecast_date)
                daily_forecast['date'] = forecast_date.strftime('%Y-%m-%d')
                forecast_data.append(daily_forecast)
            return pd.DataFrame(forecast_data)
        
        lat, lon = coordinates
        
        # URL de l'API pour les prévisions
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/next{days}days?key={self.api_key}&unitGroup=metric&include=days"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            forecast_data = []
            for day in data.get('days', [])[:days]:
                date_str = day.get('datetime')
                
                # Déterminer la condition de course
                precip = day.get('precip', 0)
                if precip > 5:
                    racing_condition = 'very_wet'
                elif precip > 2:
                    racing_condition = 'wet'
                elif precip > 0.5:
                    racing_condition = 'damp'
                else:
                    racing_condition = 'dry'
                
                daily_forecast = {
                    'date': date_str,
                    'temp_celsius': day.get('temp'),
                    'temp_min': day.get('tempmin'),
                    'temp_max': day.get('tempmax'),
                    'rain_mm': day.get('precip', 0),
                    'rain_probability': day.get('precipprob', 0) / 100 if 'precipprob' in day else 0,
                    'wind_speed_ms': day.get('windspeed') * 0.44704 if 'windspeed' in day else 0,  # mph à m/s
                    'humidity': day.get('humidity'),
                    'cloud_cover': day.get('cloudcover'),
                    'conditions': day.get('conditions'),
                    'weather_is_dry': racing_condition == 'dry',
                    'weather_is_any_wet': racing_condition != 'dry',
                    'weather_is_very_wet': racing_condition == 'very_wet',
                    'weather_temp_mild': 15 <= day.get('temp', 22) <= 25,
                    'weather_temp_hot': day.get('temp', 22) > 25,
                    'weather_high_wind': (day.get('windspeed', 0) * 0.44704) > 8,
                    'racing_condition': racing_condition
                }
                
                forecast_data.append(daily_forecast)
            
            return pd.DataFrame(forecast_data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des prévisions météo: {str(e)}")
            # Utiliser des données simulées en cas d'erreur
            forecast_data = []
            start_date = datetime.now()
            for i in range(days):
                forecast_date = start_date + timedelta(days=i)
                daily_forecast = self.get_simulated_weather(circuit, forecast_date)
                daily_forecast['date'] = forecast_date.strftime('%Y-%m-%d')
                forecast_data.append(daily_forecast)
            return pd.DataFrame(forecast_data)
    
    def get_weather_for_race_day(self, circuit: str, race_date) -> pd.DataFrame:
        """
        Récupère les prévisions météo pour le jour de course.
        
        Args:
            circuit: Nom ou identifiant du circuit
            race_date: Date de la course (string YYYY-MM-DD ou datetime)
            
        Returns:
            DataFrame avec les prévisions météo pour la journée de course
        """
        # Convertir la date si nécessaire
        if isinstance(race_date, str):
            race_date = datetime.strptime(race_date, '%Y-%m-%d')
        
        date_str = race_date.strftime('%Y-%m-%d')
        
        # Récupérer les coordonnées du circuit
        coordinates = self.get_circuit_coordinates(circuit)
        
        if not coordinates or not self.api_key:
            # Si pas de coordonnées ou pas de clé API, utiliser la simulation
            simulated_data = self.get_simulated_weather(circuit, race_date)
            simulated_data['date'] = date_str
            return pd.DataFrame([simulated_data])
        
        lat, lon = coordinates
        
        # URL de l'API pour la prévision du jour spécifique
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}?key={self.api_key}&unitGroup=metric&include=hours"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            hourly_data = []
            day_data = data.get('days', [{}])[0]
            
            # Récupérer les données horaires pour la journée de course (focus sur les heures de course 12-18h)
            for hour_data in day_data.get('hours', []):
                hour = int(hour_data.get('datetime', '00:00:00').split(':')[0])
                
                # Determiner la condition de course pour cette heure
                precip = hour_data.get('precip', 0)
                if precip > 5:
                    racing_condition = 'very_wet'
                elif precip > 2:
                    racing_condition = 'wet'
                elif precip > 0.5:
                    racing_condition = 'damp'
                else:
                    racing_condition = 'dry'
                
                hourly_forecast = {
                    'date': date_str,
                    'hour': hour,
                    'temp_celsius': hour_data.get('temp'),
                    'rain_1h_mm': hour_data.get('precip', 0),
                    'wind_speed_ms': hour_data.get('windspeed') * 0.44704 if 'windspeed' in hour_data else 0,
                    'humidity': hour_data.get('humidity'),
                    'cloud_cover': hour_data.get('cloudcover'),
                    'conditions': hour_data.get('conditions'),
                    'weather_is_dry': racing_condition == 'dry',
                    'weather_is_any_wet': racing_condition != 'dry',
                    'weather_is_very_wet': racing_condition == 'very_wet',
                    'weather_temp_mild': 15 <= hour_data.get('temp', 22) <= 25,
                    'weather_temp_hot': hour_data.get('temp', 22) > 25,
                    'weather_high_wind': (hour_data.get('windspeed', 0) * 0.44704) > 8,
                    'racing_condition': racing_condition
                }
                
                hourly_data.append(hourly_forecast)
            
            return pd.DataFrame(hourly_data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données météo pour la course: {str(e)}")
            # Utiliser des données simulées en cas d'erreur
            simulated_data = self.get_simulated_weather(circuit, race_date)
            # Créer des données horaires simulées
            hourly_data = []
            for hour in range(8, 21):  # Heures de la journée pertinentes pour une course
                hourly_forecast = simulated_data.copy()
                hourly_forecast['date'] = date_str
                hourly_forecast['hour'] = hour
                # Ajuster légèrement la température selon l'heure
                hourly_forecast['temp_celsius'] += (hour - 14) * 0.5  # Plus chaud à 14h
                hourly_data.append(hourly_forecast)
            return pd.DataFrame(hourly_data)
    
    def get_historical_weather(self, circuit: str, date) -> pd.DataFrame:
        """
        Récupère les données météo historiques pour un circuit et une date.
        
        Args:
            circuit: Nom ou identifiant du circuit
            date: Date historique (string YYYY-MM-DD ou datetime)
            
        Returns:
            DataFrame avec les données météo historiques
        """
        # Convertir la date si nécessaire
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        date_str = date.strftime('%Y-%m-%d')
        
        # Récupérer les coordonnées du circuit
        coordinates = self.get_circuit_coordinates(circuit)
        
        if not coordinates or not self.api_key:
            # Si pas de coordonnées ou pas de clé API, utiliser la simulation
            simulated_data = self.get_simulated_weather(circuit, date)
            simulated_data['date'] = date_str
            return pd.DataFrame([simulated_data])
        
        lat, lon = coordinates
        
        # URL de l'API pour les données historiques
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}?key={self.api_key}&unitGroup=metric&include=hours"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            hourly_data = []
            day_data = data.get('days', [{}])[0]
            
            # Récupérer les données horaires
            for hour_data in day_data.get('hours', []):
                hour = int(hour_data.get('datetime', '00:00:00').split(':')[0])
                
                # Determiner la condition de course pour cette heure
                precip = hour_data.get('precip', 0)
                if precip > 5:
                    racing_condition = 'very_wet'
                elif precip > 2:
                    racing_condition = 'wet'
                elif precip > 0.5:
                    racing_condition = 'damp'
                else:
                    racing_condition = 'dry'
                
                hourly_forecast = {
                    'date': date_str,
                    'hour': hour,
                    'temp_celsius': hour_data.get('temp'),
                    'rain_1h_mm': hour_data.get('precip', 0),
                    'wind_speed_ms': hour_data.get('windspeed') * 0.44704 if 'windspeed' in hour_data else 0,
                    'humidity': hour_data.get('humidity'),
                    'cloud_cover': hour_data.get('cloudcover'),
                    'conditions': hour_data.get('conditions'),
                    'weather_is_dry': racing_condition == 'dry',
                    'weather_is_any_wet': racing_condition != 'dry',
                    'weather_is_very_wet': racing_condition == 'very_wet',
                    'weather_temp_mild': 15 <= hour_data.get('temp', 22) <= 25,
                    'weather_temp_hot': hour_data.get('temp', 22) > 25,
                    'weather_high_wind': (hour_data.get('windspeed', 0) * 0.44704) > 8,
                    'racing_condition': racing_condition
                }
                
                hourly_data.append(hourly_forecast)
            
            return pd.DataFrame(hourly_data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données météo historiques: {str(e)}")
            # Utiliser des données simulées en cas d'erreur
            simulated_data = self.get_simulated_weather(circuit, date)
            # Créer des données horaires simulées
            hourly_data = []
            for hour in range(8, 21):  # Heures de la journée pertinentes pour une course
                hourly_forecast = simulated_data.copy()
                hourly_forecast['date'] = date_str
                hourly_forecast['hour'] = hour
                # Ajuster légèrement la température selon l'heure
                hourly_forecast['temp_celsius'] += (hour - 14) * 0.5  # Plus chaud à 14h
                hourly_data.append(hourly_forecast)
            return pd.DataFrame(hourly_data)
    
    def get_weather_alerts(self, circuit: str) -> List[Dict[str, Any]]:
        """
        Récupère les alertes météorologiques pour un circuit.
        
        Args:
            circuit: Nom ou identifiant du circuit
            
        Returns:
            Liste de dictionnaires contenant les alertes météo
        """
        # Récupérer les coordonnées du circuit
        coordinates = self.get_circuit_coordinates(circuit)
        
        if not coordinates or not self.api_key:
            # Si pas de coordonnées ou pas de clé API, retourner une liste vide
            return []
        
        lat, lon = coordinates
        
        # URL de l'API pour les alertes
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today?key={self.api_key}&unitGroup=metric&include=alerts"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            alerts = data.get('alerts', [])
            
            formatted_alerts = []
            for alert in alerts:
                formatted_alerts.append({
                    'event': alert.get('event'),
                    'description': alert.get('description'),
                    'onset': alert.get('onset'),
                    'ends': alert.get('ends'),
                    'severity': alert.get('severity')
                })
            
            return formatted_alerts
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des alertes météo: {str(e)}")
            return []
    
    def get_weather_impact_probability(self, circuit: str) -> Dict[str, Any]:
        """
        Calcule la probabilité d'impact de la météo sur la course.
        
        Args:
            circuit: Nom ou identifiant du circuit
            
        Returns:
            Dictionnaire avec les probabilités d'impact
        """
        # Récupérer les prévisions pour le circuit
        forecast = self.get_weather_forecast(circuit, days=7)
        
        if forecast.empty:
            return {
                'rain_probability': 0.2,
                'high_wind_probability': 0.1,
                'extreme_temp_probability': 0.1,
                'weather_impact_overall': 0.15
            }
        
        # Calculer les probabilités d'impact
        rain_probability = forecast['weather_is_any_wet'].mean()
        high_wind_probability = forecast['weather_high_wind'].mean()
        
        extreme_temp_probability = (
            (forecast['temp_celsius'] < 10).mean() + 
            (forecast['temp_celsius'] > 30).mean()
        ) / 2
        
        # Impact global (combinaison pondérée des facteurs)
        weather_impact_overall = (
            0.6 * rain_probability + 
            0.3 * high_wind_probability + 
            0.1 * extreme_temp_probability
        )
        
        return {
            'rain_probability': float(rain_probability),
            'high_wind_probability': float(high_wind_probability),
            'extreme_temp_probability': float(extreme_temp_probability),
            'weather_impact_overall': float(weather_impact_overall)
        }