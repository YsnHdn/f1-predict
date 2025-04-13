"""
Weather API Client for F1 prediction project.
This module handles retrieval of weather data for F1 circuits 
using the OpenWeatherMap API and provides both historical data and forecasts.
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv

from api.cache.manager import CacheManager

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# F1 circuits coordinates (name: [lat, lon])
# These coordinates are approximations of circuit locations
F1_CIRCUITS = {
    'bahrain': [26.0325, 50.5106],
    'jeddah': [21.6319, 39.1044],
    'albert_park': [-37.8497, 144.9680],  # Melbourne
    'imola': [44.3439, 11.7167],
    'miami': [25.9581, -80.2389],
    'monaco': [43.7347, 7.4206],
    'barcelona': [41.5691, 2.2611],
    'montreal': [45.5017, -73.5673],
    'silverstone': [52.0786, -1.0169],
    'spielberg': [47.2225, 14.7607],  # Red Bull Ring
    'paul_ricard': [43.2517, 5.7917],
    'hungaroring': [47.5830, 19.2526],
    'spa': [50.4372, 5.9714],
    'zandvoort': [52.3875, 4.5429],
    'monza': [45.6156, 9.2811],
    'baku': [40.3725, 49.8533],
    'marina_bay': [1.2914, 103.8639],  # Singapore
    'suzuka': [34.8431, 136.5407],
    'americas': [30.1328, -97.6411],  # COTA
    'rodriguez': [19.4042, -99.0907],  # Mexico City
    'interlagos': [-23.7036, -46.6997],  # São Paulo
    'las_vegas': [36.1164, -115.1694],
    'losail': [25.4661, 51.4528],  # Qatar
    'yas_marina': [24.4672, 54.6031],  # Abu Dhabi
    'shanghai': [31.3389, 121.2208],
    'portimao': [37.2306, -8.6267],
    'istanbul': [40.9517, 29.4100],
    'sochi': [43.4057, 39.9578],
    'nurburgring': [50.3356, 6.9475],
    'mugello': [43.9975, 11.3719],
    'sakhir': [26.0325, 50.5106],  # Bahrain outer circuit
    'portugal': [37.2306, -8.6267],  # Same as Portimao
    'saudi_arabia': [21.6319, 39.1044],  # Same as Jeddah
    'qatar': [25.4661, 51.4528],  # Same as Losail
    'austria': [47.2225, 14.7607],  # Same as Red Bull Ring
    'styria': [47.2225, 14.7607],  # Same as Red Bull Ring
    'emilia_romagna': [44.3439, 11.7167],  # Same as Imola
    'united_states': [30.1328, -97.6411],  # Same as COTA
}

# Weather condition codes mapping to more usable categories
WEATHER_CODES = {
    # Thunderstorm
    'thunderstorm_light': [200, 210, 230],
    'thunderstorm_moderate': [201, 211, 231],
    'thunderstorm_heavy': [202, 212, 221, 232],
    
    # Drizzle
    'drizzle_light': [300, 310],
    'drizzle_moderate': [301, 311, 313, 321],
    'drizzle_heavy': [302, 312, 314],
    
    # Rain
    'rain_light': [500, 520],
    'rain_moderate': [501, 521, 531],
    'rain_heavy': [502, 503, 504, 522],
    'rain_freezing': [511],
    
    # Snow
    'snow_light': [600, 612, 615, 620],
    'snow_moderate': [601, 613, 616, 621],
    'snow_heavy': [602, 622],
    'snow_sleet': [611, 511],
    'snow_shower': [621, 622],
    
    # Atmosphere
    'mist': [701],
    'smoke': [711],
    'haze': [721],
    'dust': [731, 761],
    'fog': [741],
    'sand': [751],
    'ash': [762],
    'squall': [771],
    'tornado': [781],
    
    # Clouds
    'clear': [800],
    'clouds_few': [801],
    'clouds_scattered': [802],
    'clouds_broken': [803],
    'clouds_overcast': [804],
}

# Simplified categorization for racing conditions
RACING_CONDITIONS = {
    'dry': ['clear', 'clouds_few', 'clouds_scattered', 'clouds_broken', 'clouds_overcast', 
            'mist', 'smoke', 'haze', 'dust', 'sand'],
    'damp': ['drizzle_light', 'drizzle_moderate'],
    'wet': ['drizzle_heavy', 'rain_light', 'rain_moderate'],
    'very_wet': ['rain_heavy', 'rain_freezing', 'thunderstorm_light', 'thunderstorm_moderate', 'thunderstorm_heavy'],
    'snow': ['snow_light', 'snow_moderate', 'snow_heavy', 'snow_sleet', 'snow_shower'],
    'low_visibility': ['fog', 'ash', 'squall', 'tornado']
}

class WeatherClient:
    """
    Client for interacting with weather APIs to retrieve weather data for F1 circuits.
    Handles caching and data formatting.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize the Weather API client.
        
        Args:
            api_key: Optional API key for weather service. If None, uses WEATHER_API_KEY env var.
            cache_dir: Optional directory for cache. If None, default cache location is used.
        """
        # Set API key
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        if not self.api_key:
            logger.warning("No Weather API key provided. API calls will fail.")
        
        # Base URLs for OpenWeatherMap API
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.geo_url = "https://api.openweathermap.org/geo/1.0"
        self.onecall_url = f"{self.base_url}/onecall"
        self.historical_url = f"{self.onecall_url}/timemachine"
        self.forecast_url = f"{self.base_url}/forecast"
        
        # Initialize cache manager
        self.cache_manager = CacheManager('weather')
        
        logger.info("Weather client initialized")
    
    def _get_weather_code_category(self, code: int) -> str:
        """
        Map OpenWeatherMap weather code to a category.
        
        Args:
            code: Weather condition code from API
        
        Returns:
            String category of the weather condition
        """
        for category, codes in WEATHER_CODES.items():
            if code in codes:
                return category
        return "unknown"
    
    def _get_racing_condition(self, weather_category: str) -> str:
        """
        Map weather category to racing condition.
        
        Args:
            weather_category: Weather category from _get_weather_code_category
        
        Returns:
            String describing racing condition (dry, damp, wet, etc.)
        """
        for condition, categories in RACING_CONDITIONS.items():
            if weather_category in categories:
                return condition
        return "unknown"
    
    def _get_circuit_coordinates(self, circuit_name: str) -> Tuple[float, float]:
        """
        Get coordinates for an F1 circuit.
        
        Args:
            circuit_name: Circuit identifier (lowercase, underscores)
        
        Returns:
            Tuple of (latitude, longitude)
        
        Raises:
            ValueError: If circuit is not found
        """
        circuit_key = circuit_name.lower().replace(' ', '_')
        if circuit_key in F1_CIRCUITS:
            return F1_CIRCUITS[circuit_key]
        
        # Try to find a partial match
        for key in F1_CIRCUITS:
            if circuit_key in key or key in circuit_key:
                logger.info(f"Using coordinates for {key} for circuit {circuit_name}")
                return F1_CIRCUITS[key]
        
        raise ValueError(f"Circuit {circuit_name} not found in coordinates database")
    
    def _kelvin_to_celsius(self, kelvin: float) -> float:
        """Convert temperature from Kelvin to Celsius."""
        return round(kelvin - 273.15, 2)
    
    def _format_weather_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Format raw weather API response into a pandas DataFrame.
        
        Args:
            data: Raw API response data
        
        Returns:
            DataFrame with formatted weather data
        """
        # Initialize an empty list to hold the formatted data
        formatted_data = []
        
        # Handle different API response formats
        if 'hourly' in data:
            # One Call API format
            weather_items = data['hourly']
        elif 'list' in data:
            # Forecast API format
            weather_items = data['list']
        else:
            # Unknown format
            logger.error("Unknown weather data format")
            return pd.DataFrame()
        
        # Process each weather data point
        for item in weather_items:
            # Get timestamp
            if 'dt' in item:
                timestamp = datetime.fromtimestamp(item['dt'])
            else:
                continue
            
            # Extract main weather data
            main_data = item.get('main', {})
            temp = self._kelvin_to_celsius(main_data.get('temp', 0))
            feels_like = self._kelvin_to_celsius(main_data.get('feels_like', 0))
            pressure = main_data.get('pressure', 0)
            humidity = main_data.get('humidity', 0)
            
            # Extract wind data
            wind_data = item.get('wind', {})
            wind_speed = wind_data.get('speed', 0)  # m/s
            wind_direction = wind_data.get('deg', 0)
            wind_gust = wind_data.get('gust', wind_speed)  # Use speed if gust not available
            
            # Extract precipitation data
            rain_1h = item.get('rain', {}).get('1h', 0) if 'rain' in item else 0
            snow_1h = item.get('snow', {}).get('1h', 0) if 'snow' in item else 0
            
            # Extract weather condition
            weather_info = item.get('weather', [{}])[0] if item.get('weather') else {}
            weather_id = weather_info.get('id', 800)  # Default to clear sky
            weather_main = weather_info.get('main', 'Clear')
            weather_description = weather_info.get('description', 'clear sky')
            
            # Get weather category and racing condition
            weather_category = self._get_weather_code_category(weather_id)
            racing_condition = self._get_racing_condition(weather_category)
            
            # Additional data
            clouds = item.get('clouds', {}).get('all', 0) if 'clouds' in item else 0
            visibility = item.get('visibility', 10000)  # Default to 10km
            
            # Create a dictionary for this data point
            data_point = {
                'timestamp': timestamp,
                'temp_celsius': temp,
                'feels_like_celsius': feels_like,
                'pressure_hpa': pressure,
                'humidity_percent': humidity,
                'wind_speed_ms': wind_speed,
                'wind_direction_degrees': wind_direction,
                'wind_gust_ms': wind_gust,
                'rain_1h_mm': rain_1h,
                'snow_1h_mm': snow_1h,
                'clouds_percent': clouds,
                'visibility_meters': visibility,
                'weather_id': weather_id,
                'weather_main': weather_main,
                'weather_description': weather_description,
                'weather_category': weather_category,
                'racing_condition': racing_condition
            }
            
            formatted_data.append(data_point)
        
        # Create DataFrame
        df = pd.DataFrame(formatted_data)
        
        # Add location data if available
        if 'lat' in data and 'lon' in data:
            df['latitude'] = data['lat']
            df['longitude'] = data['lon']
        
        # Add timezone data if available
        if 'timezone' in data:
            df['timezone'] = data['timezone']
        
        return df
    
    def get_current_weather(self, circuit_name: str) -> pd.DataFrame:
        """
        Get current weather for an F1 circuit.
        
        Args:
            circuit_name: Circuit identifier
        
        Returns:
            DataFrame with current weather data
        """
        cache_key = f"current_{circuit_name.lower()}"
        cached_data = self.cache_manager.get(cache_key)
        
        # Only return cached data if it's from the last hour
        if cached_data is not None:
            age = (datetime.now() - cached_data['timestamp']).total_seconds() / 3600
            if age < 1:
                logger.info(f"Using cached current weather for {circuit_name} (age: {age:.2f} hours)")
                return cached_data['data']
        
        try:
            # Get circuit coordinates
            lat, lon = self._get_circuit_coordinates(circuit_name)
            
            # Make API request
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'standard'  # Kelvin for temperature
            }
            
            response = requests.get(f"{self.base_url}/weather", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the data
            weather_df = self._format_weather_data({'list': [data]})
            
            # Add circuit information
            weather_df['circuit'] = circuit_name
            
            # Cache the data with timestamp
            cache_data = {
                'timestamp': datetime.now(),
                'data': weather_df
            }
            self.cache_manager.set(cache_key, cache_data, expiry=3600)  # 1 hour
            
            return weather_df
            
        except Exception as e:
            logger.error(f"Error getting current weather for {circuit_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_weather_forecast(self, circuit_name: str, days: int = 5) -> pd.DataFrame:
        """
        Get weather forecast for an F1 circuit.
        
        Args:
            circuit_name: Circuit identifier
            days: Number of days to forecast (max 5)
        
        Returns:
            DataFrame with forecast weather data
        """
        days = min(days, 5)  # Limit to 5 days (free API limit)
        cache_key = f"forecast_{circuit_name.lower()}_{days}"
        cached_data = self.cache_manager.get(cache_key)
        
        # Only return cached data if it's from the last 3 hours
        if cached_data is not None:
            age = (datetime.now() - cached_data['timestamp']).total_seconds() / 3600
            if age < 3:
                logger.info(f"Using cached forecast for {circuit_name} (age: {age:.2f} hours)")
                return cached_data['data']
        
        try:
            # Get circuit coordinates
            lat, lon = self._get_circuit_coordinates(circuit_name)
            
            # Make API request
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'standard',  # Kelvin for temperature
                'cnt': min(days * 8, 40)  # 8 data points per day (3-hour intervals)
            }
            
            response = requests.get(self.forecast_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the data
            weather_df = self._format_weather_data(data)
            
            # Add circuit information
            weather_df['circuit'] = circuit_name
            
            # Cache the data with timestamp
            cache_data = {
                'timestamp': datetime.now(),
                'data': weather_df
            }
            self.cache_manager.set(cache_key, cache_data, expiry=10800)  # 3 hours
            
            return weather_df
            
        except Exception as e:
            logger.error(f"Error getting forecast for {circuit_name}: {str(e)}")
            return pd.DataFrame()
    
    def get_historical_weather(self, circuit_name: str, date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get historical weather for an F1 circuit for a specific date.
        
        Args:
            circuit_name: Circuit identifier
            date: Date to get weather for (string 'YYYY-MM-DD' or datetime)
        
        Returns:
            DataFrame with historical weather data
        """
        # Convert string date to datetime if needed
        if isinstance(date, str):
            try:
                date = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format. Please use 'YYYY-MM-DD': {date}")
                return pd.DataFrame()
        
        # Format date for cache key
        date_str = date.strftime('%Y-%m-%d')
        cache_key = f"historical_{circuit_name.lower()}_{date_str}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Using cached historical weather for {circuit_name} on {date_str}")
            return cached_data
        
        try:
            # Get circuit coordinates
            lat, lon = self._get_circuit_coordinates(circuit_name)
            
            # Convert date to unix timestamp
            timestamp = int(date.timestamp())
            
            # Make API request
            params = {
                'lat': lat,
                'lon': lon,
                'dt': timestamp,
                'appid': self.api_key,
                'units': 'standard'  # Kelvin for temperature
            }
            
            response = requests.get(self.historical_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the data
            weather_df = self._format_weather_data(data)
            
            # Add circuit information
            weather_df['circuit'] = circuit_name
            weather_df['date'] = date_str
            
            # Cache the data (historical data doesn't change, so long expiry)
            self.cache_manager.set(cache_key, weather_df, expiry=30*86400)  # 30 days
            
            return weather_df
            
        except Exception as e:
            logger.error(f"Error getting historical weather for {circuit_name} on {date_str}: {str(e)}")
            return pd.DataFrame()
    
    def get_race_weekend_forecast(self, circuit_name: str, weekend_start: Union[str, datetime]) -> Dict[str, pd.DataFrame]:
        """
        Get weather forecast for an entire F1 race weekend.
        
        Args:
            circuit_name: Circuit identifier
            weekend_start: Start date of the race weekend (Friday)
        
        Returns:
            Dictionary with forecast for each day of the weekend
        """
        # Convert string date to datetime if needed
        if isinstance(weekend_start, str):
            try:
                weekend_start = datetime.strptime(weekend_start, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format. Please use 'YYYY-MM-DD': {weekend_start}")
                return {}
        
        # Ensure weekend_start is a Friday
        day_of_week = weekend_start.weekday()
        if day_of_week != 4:  # 4 is Friday in Python's weekday
            logger.warning(f"Weekend start date {weekend_start.strftime('%Y-%m-%d')} is not a Friday")
            # Adjust to previous or next Friday if needed
        
        # Get 3-day forecast (Friday, Saturday, Sunday)
        forecast_df = self.get_weather_forecast(circuit_name, days=3)
        
        if forecast_df.empty:
            return {}
        
        # Split forecast by day
        result = {}
        
        # Calculate date range for the weekend
        weekend_dates = [
            (weekend_start + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(3)
        ]
        
        session_names = ['Practice', 'Qualifying', 'Race']
        
        for i, date_str in enumerate(weekend_dates):
            # Filter forecast for this day
            day_forecast = forecast_df[forecast_df['timestamp'].dt.strftime('%Y-%m-%d') == date_str]
            
            if not day_forecast.empty:
                result[session_names[i]] = day_forecast
            else:
                logger.warning(f"No forecast data available for {date_str}")
        
        return result
    
    def get_weather_statistics(self, circuit_name: str, start_date: Union[str, datetime], 
                              end_date: Union[str, datetime]) -> Dict[str, Any]:
        """
        Get weather statistics for a circuit over a date range.
        
        Args:
            circuit_name: Circuit identifier
            start_date: Start date (string 'YYYY-MM-DD' or datetime)
            end_date: End date (string 'YYYY-MM-DD' or datetime)
        
        Returns:
            Dictionary with weather statistics
        """
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Format dates for cache key
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        cache_key = f"stats_{circuit_name.lower()}_{start_str}_{end_str}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Using cached weather statistics for {circuit_name}")
            return cached_data
        
        # Generate date range
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date)
            current_date += timedelta(days=1)
        
        # Collect weather data for each date
        all_weather_data = []
        for date in date_range:
            weather_df = self.get_historical_weather(circuit_name, date)
            if not weather_df.empty:
                all_weather_data.append(weather_df)
        
        # Combine all data
        if not all_weather_data:
            logger.warning(f"No weather data found for {circuit_name} in date range")
            return {}
        
        combined_df = pd.concat(all_weather_data, ignore_index=True)
        
        # Calculate statistics
        stats = {
            'circuit': circuit_name,
            'start_date': start_str,
            'end_date': end_str,
            'days_analyzed': len(all_weather_data),
            'temperature': {
                'mean': combined_df['temp_celsius'].mean(),
                'min': combined_df['temp_celsius'].min(),
                'max': combined_df['temp_celsius'].max(),
                'std': combined_df['temp_celsius'].std()
            },
            'precipitation': {
                'mean': combined_df['rain_1h_mm'].mean(),
                'max': combined_df['rain_1h_mm'].max(),
                'rainy_hours': (combined_df['rain_1h_mm'] > 0).sum()
            },
            'wind': {
                'mean_speed': combined_df['wind_speed_ms'].mean(),
                'max_speed': combined_df['wind_speed_ms'].max(),
                'mean_gust': combined_df['wind_gust_ms'].mean(),
                'max_gust': combined_df['wind_gust_ms'].max()
            },
            'conditions': {
                condition: (combined_df['racing_condition'] == condition).sum()
                for condition in ['dry', 'damp', 'wet', 'very_wet', 'snow', 'low_visibility']
            }
        }
        
        # Cache the statistics
        self.cache_manager.set(cache_key, stats, expiry=7*86400)  # 7 days
        
        return stats
    
    def get_weather_impact_probability(self, circuit_name: str) -> Dict[str, float]:
        """
        Calculate probability of different weather impacts at a circuit based on historical data.
        
        Args:
            circuit_name: Circuit identifier
        
        Returns:
            Dictionary with probabilities of different weather impacts
        """
        cache_key = f"impact_prob_{circuit_name.lower()}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Using cached weather impact probabilities for {circuit_name}")
            return cached_data
        
        # Try to get historical race date for this circuit
        # This would require additional data about when races are typically held
        # For now, we'll use a simple approach with the last 5 years
        
        current_year = datetime.now().year
        race_months = {
            'bahrain': 3,
            'jeddah': 3,
            'albert_park': 4,
            'imola': 4,
            'miami': 5,
            'monaco': 5,
            'barcelona': 6,
            'montreal': 6,
            'silverstone': 7,
            'spielberg': 7,
            'paul_ricard': 7,
            'hungaroring': 7,
            'spa': 8,
            'zandvoort': 9,
            'monza': 9,
            'baku': 6,
            'marina_bay': 10,
            'suzuka': 10,
            'americas': 10,
            'rodriguez': 10,
            'interlagos': 11,
            'las_vegas': 11,
            'losail': 11,
            'yas_marina': 12,
        }
        
        circuit_key = circuit_name.lower().replace(' ', '_')
        race_month = race_months.get(circuit_key, datetime.now().month)
        
        # Analyze data from the last 5 years
        years_to_analyze = 5
        all_stats = []
        
        for year in range(current_year - years_to_analyze, current_year):
            # Create a date range around the typical race month (2 weeks)
            race_date = datetime(year, race_month, 15)
            start_date = race_date - timedelta(days=7)
            end_date = race_date + timedelta(days=7)
            
            stats = self.get_weather_statistics(circuit_name, start_date, end_date)
            if stats:
                all_stats.append(stats)
        
        if not all_stats:
            logger.warning(f"No historical data found for {circuit_name}")
            return {}
        
        # Calculate probabilities
        total_hours = sum(stat['days_analyzed'] * 24 for stat in all_stats)
        if total_hours == 0:
            return {}
        
        # Sum condition counts across all years
        condition_counts = {
            condition: sum(stat['conditions'].get(condition, 0) for stat in all_stats)
            for condition in ['dry', 'damp', 'wet', 'very_wet', 'snow', 'low_visibility']
        }
        
        # Calculate probabilities
        probabilities = {
            f"{condition}_probability": count / total_hours
            for condition, count in condition_counts.items()
        }
        
        # Add other impact measures
        rain_hours = sum(stat['precipitation']['rainy_hours'] for stat in all_stats)
        probabilities['rain_probability'] = rain_hours / total_hours
        
        # High wind probability (above 8 m/s)
        high_wind_hours = sum(
            (stat.get('wind', {}).get('mean_speed', 0) > 8).sum()
            for stat in all_stats
        )
        probabilities['high_wind_probability'] = high_wind_hours / total_hours
        
        # Temperature extremes
        extreme_heat_hours = sum(
            (stat.get('temperature', {}).get('max', 0) > 30).sum()
            for stat in all_stats
        )
        probabilities['extreme_heat_probability'] = extreme_heat_hours / total_hours
        
        # Cache the probabilities
        self.cache_manager.set(cache_key, probabilities, expiry=30*86400)  # 30 days
        
        return probabilities
    
    def analyze_weather_trends(self, circuit_name: str, years: int = 5) -> Dict[str, Any]:
        """
        Analyze weather trends at a circuit over multiple years.
        
        Args:
            circuit_name: Circuit identifier
            years: Number of years to analyze
        
        Returns:
            Dictionary with weather trends
        """
        cache_key = f"trends_{circuit_name.lower()}_{years}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Using cached weather trends for {circuit_name}")
            return cached_data
        
        # This would require more historical data than the free API provides
        # For a real implementation, you would need a premium weather API
        # or historical weather dataset
        
        # Return placeholder for now
        trends = {
            'circuit': circuit_name,
            'years_analyzed': years,
            'note': 'Full weather trend analysis requires premium API access',
            'limited_trends': {
                'temperature_trend': 'stable',  # placeholder
                'precipitation_trend': 'increasing',  # placeholder
                'extreme_events_trend': 'stable'  # placeholder
            }
        }
        
        # Cache the trends
        self.cache_manager.set(cache_key, trends, expiry=30*86400)  # 30 days
        
        return trends
    
    def get_weather_for_race_day(self, circuit_name: str, race_date: Union[str, datetime]) -> pd.DataFrame:
        """
        Get detailed weather forecast for race day.
        
        Args:
            circuit_name: Circuit identifier
            race_date: Race date (string 'YYYY-MM-DD' or datetime)
        
        Returns:
            DataFrame with hourly weather forecast for race day
        """
        # Convert string date to datetime if needed
        if isinstance(race_date, str):
            try:
                race_date = datetime.strptime(race_date, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Invalid date format. Please use 'YYYY-MM-DD': {race_date}")
                return pd.DataFrame()
        
        # Format date for cache key
        date_str = race_date.strftime('%Y-%m-%d')
        cache_key = f"race_day_{circuit_name.lower()}_{date_str}"
        cached_data = self.cache_manager.get(cache_key)
        
        # Only return cached data if it's from the last 3 hours and race is in the future
        if cached_data is not None and race_date > datetime.now():
            age = (datetime.now() - cached_data['timestamp']).total_seconds() / 3600
            if age < 3:
                logger.info(f"Using cached race day forecast for {circuit_name} (age: {age:.2f} hours)")
                return cached_data['data']
        
        # Get the forecast for the whole weekend
        forecast_df = self.get_weather_forecast(circuit_name, days=5)
        
        if forecast_df.empty:
            return pd.DataFrame()
        
        # Filter for race day
        race_day_forecast = forecast_df[forecast_df['timestamp'].dt.strftime('%Y-%m-%d') == date_str]
        
        if race_day_forecast.empty:
            logger.warning(f"No forecast data available for race day {date_str}")
            return pd.DataFrame()
        
        # Add F1 specific context
        race_day_forecast['event'] = 'Race'
        race_day_forecast['circuit_name'] = circuit_name
        
        # Add hour of day for easier filtering
        race_day_forecast['hour'] = race_day_forecast['timestamp'].dt.hour
        
        # Sort by time
        race_day_forecast = race_day_forecast.sort_values('timestamp')
        
        # Add tire recommendation based on weather
        race_day_forecast['recommended_tire'] = race_day_forecast['racing_condition'].map({
            'dry': 'Slick',
            'damp': 'Intermediate',
            'wet': 'Intermediate',
            'very_wet': 'Wet',
            'snow': 'Wet',
            'low_visibility': 'Depends on precipitation'
        })
        
        # Cache the data with timestamp if race is in the future
        if race_date > datetime.now():
            cache_data = {
                'timestamp': datetime.now(),
                'data': race_day_forecast
            }
            self.cache_manager.set(cache_key, cache_data, expiry=10800)  # 3 hours
        
        return race_day_forecast
    
    def compare_circuit_weather(self, circuit_names: List[str]) -> Dict[str, Any]:
        """
        Compare weather characteristics between multiple circuits.
        
        Args:
            circuit_names: List of circuit identifiers to compare
        
        Returns:
            Dictionary with comparative weather statistics
        """
        # Create a unique cache key from sorted circuit names
        sorted_names = sorted(circuit_names)
        cache_key = f"compare_{'_'.join(sorted_names)}"
        cached_data = self.cache_manager.get(cache_key)
        
        if cached_data is not None:
            logger.info(f"Using cached circuit comparison for {', '.join(circuit_names)}")
            return cached_data
        
        # Get current conditions for each circuit
        current_conditions = {}
        for circuit in circuit_names:
            weather_df = self.get_current_weather(circuit)
            if not weather_df.empty:
                current_conditions[circuit] = {
                    'temperature': weather_df['temp_celsius'].iloc[0],
                    'condition': weather_df['racing_condition'].iloc[0],
                    'wind_speed': weather_df['wind_speed_ms'].iloc[0],
                    'precipitation': weather_df['rain_1h_mm'].iloc[0]
                }
        
        # Get historical statistics for each circuit (simplified)
        # For a full implementation, we would need more historical data
        
        # Return comparison data
        comparison = {
            'circuits_compared': circuit_names,
            'current_conditions': current_conditions,
            'temperature_ranking': sorted(current_conditions.keys(), 
                                         key=lambda x: current_conditions[x]['temperature'], 
                                         reverse=True),
            'precipitation_ranking': sorted(current_conditions.keys(), 
                                           key=lambda x: current_conditions[x]['precipitation'], 
                                           reverse=True),
            'wind_ranking': sorted(current_conditions.keys(), 
                                  key=lambda x: current_conditions[x]['wind_speed'], 
                                  reverse=True)
        }
        
        # Cache the comparison
        self.cache_manager.set(cache_key, comparison, expiry=3600)  # 1 hour
        
        return comparison
    
    def get_weather_alerts(self, circuit_name: str) -> List[Dict[str, Any]]:
        """
        Get weather alerts/warnings for a circuit.
        
        Args:
            circuit_name: Circuit identifier
        
        Returns:
            List of weather alerts
        """
        cache_key = f"alerts_{circuit_name.lower()}"
        cached_data = self.cache_manager.get(cache_key)
        
        # Only return cached data if it's from the last hour
        if cached_data is not None:
            age = (datetime.now() - cached_data['timestamp']).total_seconds() / 3600
            if age < 1:
                logger.info(f"Using cached weather alerts for {circuit_name} (age: {age:.2f} hours)")
                return cached_data['data']
        
        try:
            # Get circuit coordinates
            lat, lon = self._get_circuit_coordinates(circuit_name)
            
            # Make API request - use OneCall API which includes alerts
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'standard',
                'exclude': 'minutely,daily'  # Exclude unnecessary data
            }
            
            response = requests.get(f"{self.onecall_url}", params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract alerts
            alerts = data.get('alerts', [])
            
            # Format alerts
            formatted_alerts = []
            for alert in alerts:
                formatted_alert = {
                    'sender': alert.get('sender_name', 'Unknown'),
                    'event': alert.get('event', 'Weather alert'),
                    'start': datetime.fromtimestamp(alert.get('start', 0)),
                    'end': datetime.fromtimestamp(alert.get('end', 0)),
                    'description': alert.get('description', ''),
                    'severity': 'high' if 'warning' in alert.get('event', '').lower() else 'moderate'
                }
                formatted_alerts.append(formatted_alert)
            
            # Add race impact assessment
            for alert in formatted_alerts:
                # Assess potential race impact
                impact = 'low'
                event = alert['event'].lower()
                
                if any(term in event for term in ['thunderstorm', 'tornado', 'hurricane', 'flood']):
                    impact = 'high'
                elif any(term in event for term in ['rain', 'wind', 'snow', 'fog', 'advisory']):
                    impact = 'medium'
                
                alert['race_impact'] = impact
            
            # Cache the alerts with timestamp
            cache_data = {
                'timestamp': datetime.now(),
                'data': formatted_alerts
            }
            self.cache_manager.set(cache_key, cache_data, expiry=3600)  # 1 hour
            
            return formatted_alerts
            
        except Exception as e:
            logger.error(f"Error getting weather alerts for {circuit_name}: {str(e)}")
            return []