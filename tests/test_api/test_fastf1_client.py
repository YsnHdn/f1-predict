"""
Tests for the FastF1 API client.
"""

import os
import pytest
import pandas as pd
import fastf1
from unittest.mock import patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

from api.fastf1_client import FastF1Client
from api.cache.manager import CacheManager


class TestFastF1Client:
    """Test suite for FastF1Client."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        mock_manager = MagicMock(spec=CacheManager)
        mock_manager.get.return_value = None  # Default: no cache hit
        return mock_manager
    
    @pytest.fixture
    def client(self, mock_cache_manager):
        """Create a FastF1Client instance with mocked cache manager."""
        with patch('api.fastf1_client.CacheManager', return_value=mock_cache_manager):
            with patch('fastf1.Cache.enable_cache'):
                with patch('fastf1.plotting.setup_mpl'):
                    client = FastF1Client()
                    return client
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock FastF1 session."""
        mock = MagicMock()
        mock.event = {
            'EventName': 'TestGP',
            'OfficialEventName': 'Test Grand Prix',
            'EventDate': '2023-01-01'
        }
        # Set up mock results
        mock.results = pd.DataFrame({
            'DriverNumber': [44, 33, 16],
            'Driver': ['HAM', 'VER', 'LEC'],
            'Position': [1, 2, 3],
            'Points': [25, 18, 15]
        })
        # Set up mock laps
        mock.laps = pd.DataFrame({
            'Driver': ['HAM', 'VER', 'LEC'] * 3,
            'LapNumber': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'LapTime': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91), pd.Timedelta(seconds=92),
                        pd.Timedelta(seconds=89), pd.Timedelta(seconds=90), pd.Timedelta(seconds=91),
                        pd.Timedelta(seconds=88), pd.Timedelta(seconds=89), pd.Timedelta(seconds=90)],
            'Compound': ['SOFT', 'MEDIUM', 'HARD', 'SOFT', 'MEDIUM', 'HARD', 'SOFT', 'MEDIUM', 'HARD'],
            'Stint': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'TyreLife': [1, 1, 1, 2, 2, 2, 3, 3, 3]
        })
        # Set up mock weather data
        mock.weather_data = pd.DataFrame({
            'Time': [pd.Timestamp('2023-01-01 14:00:00'), pd.Timestamp('2023-01-01 14:15:00')],
            'AirTemp': [25.0, 26.0],
            'Humidity': [40.0, 41.0],
            'Pressure': [1013.0, 1012.5],
            'Rainfall': [0.0, 0.0],
            'TrackTemp': [45.0, 46.0],
            'WindDirection': [180.0, 185.0],
            'WindSpeed': [10.0, 12.0]
        })
        # Set up mock race control messages
        mock.race_control_messages = pd.DataFrame({
            'Time': [pd.Timestamp('2023-01-01 14:10:00'), pd.Timestamp('2023-01-01 14:30:00')],
            'Category': ['Flag', 'Flag'],
            'Message': ['YELLOW', 'CLEAR'],
            'Status': [2, 1]
        })
        
        # Mock driver info method
        mock.get_driver_info.return_value = pd.DataFrame({
            'DriverNumber': [44, 33, 16],
            'Abbreviation': ['HAM', 'VER', 'LEC'],
            'FullName': ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc'],
            'TeamName': ['Mercedes', 'Red Bull', 'Ferrari']
        })
        
        return mock
    
    def test_init(self):
        """Test initialization of FastF1Client."""
        with patch('fastf1.Cache.enable_cache') as mock_cache:
            with patch('api.fastf1_client.CacheManager'):
                with patch('fastf1.plotting.setup_mpl'):
                    # Test with default cache dir
                    client = FastF1Client()
                    mock_cache.assert_called_once()
                    
                    # Test with custom cache dir
                    mock_cache.reset_mock()
                    with TemporaryDirectory() as temp_dir:
                        client = FastF1Client(cache_dir=temp_dir)
                        mock_cache.assert_called_once()
    
    def test_get_session(self, client, mock_session, mock_cache_manager):
        """Test getting a FastF1 session."""
        with patch('fastf1.get_session', return_value=mock_session) as mock_get_session:
            # Test cache miss
            session = client.get_session(2023, 'TestGP', 'R')
            mock_get_session.assert_called_once_with(2023, 'TestGP', 'R')
            mock_cache_manager.get.assert_called_once_with('session_2023_TestGP_R')
            mock_cache_manager.set.assert_called_once_with('session_2023_TestGP_R', mock_session)
            assert session == mock_session
            
            # Test cache hit
            mock_cache_manager.reset_mock()
            mock_get_session.reset_mock()
            mock_cache_manager.get.return_value = mock_session
            
            session = client.get_session(2023, 'TestGP', 'R')
            mock_get_session.assert_not_called()
            mock_cache_manager.get.assert_called_once_with('session_2023_TestGP_R')
            assert session == mock_session
    
    def test_get_race_calendar(self, client, mock_cache_manager):
        """Test getting a race calendar."""
        mock_calendar = pd.DataFrame({
            'RoundNumber': [1, 2, 3],
            'EventName': ['Bahrain', 'Saudi Arabia', 'Australia'],
            'EventDate': ['2023-03-05', '2023-03-19', '2023-04-02']
        })
        
        with patch('fastf1.get_event_schedule', return_value=mock_calendar) as mock_get_calendar:
            # Test cache miss
            calendar = client.get_race_calendar(2023)
            mock_get_calendar.assert_called_once_with(2023)
            mock_cache_manager.get.assert_called_once_with('calendar_2023')
            mock_cache_manager.set.assert_called_once_with('calendar_2023', mock_calendar)
            assert calendar.equals(mock_calendar)
            
            # Test cache hit
            mock_cache_manager.reset_mock()
            mock_get_calendar.reset_mock()
            mock_cache_manager.get.return_value = mock_calendar
            
            calendar = client.get_race_calendar(2023)
            mock_get_calendar.assert_not_called()
            mock_cache_manager.get.assert_called_once_with('calendar_2023')
            assert calendar.equals(mock_calendar)
    
    def test_get_race_results(self, client, mock_session):
        """Test getting race results."""
        with patch.object(client, 'get_session', return_value=mock_session):
            results = client.get_race_results(2023, 'TestGP')
            assert 'Year' in results.columns
            assert 'GrandPrix' in results.columns
            assert 'TrackName' in results.columns
            assert 'Date' in results.columns
            assert results['Year'].iloc[0] == 2023
            assert results['GrandPrix'].iloc[0] == 'TestGP'
    
    def test_get_qualifying_results(self, client, mock_session):
        """Test getting qualifying results."""
        with patch.object(client, 'get_session', return_value=mock_session):
            results = client.get_qualifying_results(2023, 'TestGP')
            assert 'Year' in results.columns
            assert 'GrandPrix' in results.columns
            assert 'TrackName' in results.columns
            assert 'Date' in results.columns
            assert results['Year'].iloc[0] == 2023
            assert results['GrandPrix'].iloc[0] == 'TestGP'
    
    def test_get_sprint_results(self, client, mock_session):
        """Test getting sprint results."""
        with patch.object(client, 'get_session', return_value=mock_session):
            results = client.get_sprint_results(2023, 'TestGP')
            assert 'Year' in results.columns
            assert 'GrandPrix' in results.columns
            assert 'TrackName' in results.columns
            assert 'Date' in results.columns
            assert results['Year'].iloc[0] == 2023
            assert results['GrandPrix'].iloc[0] == 'TestGP'
        
        # Test case where there is no sprint race
        with patch.object(client, 'get_session', side_effect=Exception("No sprint session")):
            results = client.get_sprint_results(2023, 'TestGP')
            assert results is None
    
    def test_get_driver_telemetry(self, client, mock_session, mock_cache_manager):
        """Test getting driver telemetry."""
        # Create mock telemetry dataframes
        car_data = pd.DataFrame({
            'Time': [0, 1, 2],
            'Speed': [0, 100, 200],
            'RPM': [0, 10000, 12000],
            'Gear': [0, 1, 2],
            'LapNumber': [1, 1, 1]
        })
        
        pos_data = pd.DataFrame({
            'Time': [0, 1, 2],
            'X': [0, 10, 20],
            'Y': [0, 10, 20],
            'Z': [0, 0, 0],
            'LapNumber': [1, 1, 1]
        })
        
        # Create a lap series that uses proper get_car_data and get_pos_data methods
        class MockLapSeries(pd.Series):
            def get_car_data(self):
                return car_data.copy()
                
            def get_pos_data(self):
                return pos_data.copy()
        
        # Create mock driver laps DataFrame with proper method
        lap_data = {'LapNumber': 1, 'Driver': 'HAM', 'LapTime': pd.Timedelta(seconds=90)}
        mock_lap = MockLapSeries(lap_data)
        
        mock_driver_laps = pd.DataFrame([lap_data])
        
        # Override iterrows to return our custom series
        orig_iterrows = mock_driver_laps.iterrows
        mock_driver_laps.iterrows = lambda: [(0, mock_lap)]
        mock_driver_laps.pick_driver = lambda driver: mock_driver_laps
        
        mock_session.laps = mock_driver_laps
        
        with patch.object(client, 'get_session', return_value=mock_session):
            # Test cache miss
            mock_cache_manager.get.return_value = None
            telemetry = client.get_driver_telemetry(2023, 'TestGP', 'R', 'HAM')
            
            assert 'laps' in telemetry
            assert 'car_data' in telemetry
            assert 'pos_data' in telemetry
            assert telemetry['car_data'] is not None
            assert telemetry['pos_data'] is not None
            assert 'LapNumber' in telemetry['car_data'].columns
            assert 'Speed' in telemetry['car_data'].columns
            assert 'X' in telemetry['pos_data'].columns
            
            # Test specific lap retrieval with cache hit
            mock_cache_result = {
                'laps': mock_driver_laps,
                'car_data': car_data,
                'pos_data': pos_data
            }
            mock_cache_manager.get.return_value = mock_cache_result
            
            telemetry = client.get_driver_telemetry(2023, 'TestGP', 'R', 'HAM', laps=1)
            assert 'car_data' in telemetry
            assert not telemetry['car_data'].empty
            
            # Test with cache hit
            mock_cache_manager.reset_mock()
            mock_cache_result = {'laps': mock_driver_laps, 'car_data': pd.DataFrame(), 'pos_data': pd.DataFrame()}
            mock_cache_manager.get.return_value = mock_cache_result
            
            telemetry = client.get_driver_telemetry(2023, 'TestGP', 'R', 'HAM')
            assert telemetry == mock_cache_result
    
    def test_get_weather_data(self, client, mock_session):
        """Test getting weather data."""
        with patch.object(client, 'get_session', return_value=mock_session):
            weather = client.get_weather_data(2023, 'TestGP', 'R')
            
            assert 'Year' in weather.columns
            assert 'GrandPrix' in weather.columns
            assert 'Session' in weather.columns
            assert 'AirTemp' in weather.columns
            assert 'TrackTemp' in weather.columns
            assert weather['Year'].iloc[0] == 2023
            assert weather['GrandPrix'].iloc[0] == 'TestGP'
            assert weather['Session'].iloc[0] == 'R'
    
    def test_get_track_status_data(self, client, mock_session, mock_cache_manager):
        """Test getting track status data."""
        with patch.object(client, 'get_session', return_value=mock_session):
            # Test cache miss
            track_status = client.get_track_status_data(2023, 'TestGP', 'R')
            
            assert 'Year' in track_status.columns
            assert 'GrandPrix' in track_status.columns
            assert 'Session' in track_status.columns
            assert 'StatusDesc' in track_status.columns
            assert track_status['Year'].iloc[0] == 2023
            assert track_status['GrandPrix'].iloc[0] == 'TestGP'
            assert track_status['Session'].iloc[0] == 'R'
            assert track_status['StatusDesc'].iloc[0] == 'Yellow Flag'
            assert track_status['StatusDesc'].iloc[1] == 'Track Clear'
            
            # Test cache hit
            mock_cache_manager.reset_mock()
            mock_cache_manager.get.return_value = track_status
            
            result = client.get_track_status_data(2023, 'TestGP', 'R')
            assert result.equals(track_status)
            mock_cache_manager.get.assert_called_once_with('track_status_2023_TestGP_R')
    
    def test_get_driver_info(self, client, mock_session, mock_cache_manager):
        """Test getting driver information."""
        mock_schedule = pd.DataFrame({
            'EventName': ['TestGP'],
            'EventDate': ['2023-01-01']
        })
        
        with patch('fastf1.get_event_schedule', return_value=mock_schedule):
            with patch.object(client, 'get_session', return_value=mock_session):
                # Test cache miss
                driver_info = client.get_driver_info(2023)
                
                assert 'DriverNumber' in driver_info.columns
                assert 'Abbreviation' in driver_info.columns
                assert 'FullName' in driver_info.columns
                assert 'TeamName' in driver_info.columns
                assert 'Season' in driver_info.columns
                assert driver_info['Season'].iloc[0] == 2023
                
                # Test cache hit
                mock_cache_manager.reset_mock()
                mock_cache_manager.get.return_value = driver_info
                
                result = client.get_driver_info(2023)
                assert result.equals(driver_info)
                mock_cache_manager.get.assert_called_once_with('drivers_2023')
    
    def test_compare_drivers_lap_time(self, client, mock_session):
        """Test comparing driver lap times."""
        # Create proper mock laps DataFrame with pick_driver method
        laps_data = pd.DataFrame({
            'Driver': ['HAM', 'VER', 'LEC'] * 3,
            'LapNumber': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'LapTime': [pd.Timedelta(seconds=90), pd.Timedelta(seconds=91), pd.Timedelta(seconds=92),
                        pd.Timedelta(seconds=89), pd.Timedelta(seconds=90), pd.Timedelta(seconds=91),
                        pd.Timedelta(seconds=88), pd.Timedelta(seconds=89), pd.Timedelta(seconds=90)]
        })
        
        # Original DataFrame shouldn't be modified, so we need to create a proper mock
        class MockLaps(pd.DataFrame):
            def pick_driver(self, driver):
                return self[self['Driver'] == driver]
        
        # Convert to our mock class
        mock_laps = MockLaps(laps_data)
        mock_session.laps = mock_laps
        
        with patch.object(client, 'get_session', return_value=mock_session):
            comparison = client.compare_drivers_lap_time(2023, 'TestGP', 'R', ['HAM', 'VER'])
            
            assert comparison is not None
            assert 'Driver' in comparison.columns
            assert 'LapNumber' in comparison.columns
            assert 'Year' in comparison.columns
            assert 'GrandPrix' in comparison.columns
            assert 'Session' in comparison.columns
            assert comparison['Year'].iloc[0] == 2023
            assert comparison['GrandPrix'].iloc[0] == 'TestGP'
            assert comparison['Session'].iloc[0] == 'R'
            
            # Verify we have data for both drivers
            drivers_in_comparison = comparison['Driver'].unique()
            assert 'HAM' in drivers_in_comparison
            assert 'VER' in drivers_in_comparison
    
    def test_get_session_lap_data(self, client, mock_session):
        """Test getting session lap data."""
        with patch.object(client, 'get_session', return_value=mock_session):
            lap_data = client.get_session_lap_data(2023, 'TestGP', 'R')
            
            assert 'Year' in lap_data.columns
            assert 'GrandPrix' in lap_data.columns
            assert 'TrackName' in lap_data.columns
            assert 'Session' in lap_data.columns
            assert 'Driver' in lap_data.columns
            assert 'LapNumber' in lap_data.columns
            assert 'LapTime' in lap_data.columns
            assert lap_data['Year'].iloc[0] == 2023
            assert lap_data['GrandPrix'].iloc[0] == 'TestGP'
            assert lap_data['Session'].iloc[0] == 'R'
    
    def test_get_fastest_laps(self, client, mock_session):
        """Test getting fastest laps."""
        with patch.object(client, 'get_session', return_value=mock_session):
            fastest_laps = client.get_fastest_laps(2023, 'TestGP', 'R', n=2)
            
            assert 'Year' in fastest_laps.columns
            assert 'GrandPrix' in fastest_laps.columns
            assert 'TrackName' in fastest_laps.columns
            assert 'Session' in fastest_laps.columns
            assert 'Driver' in fastest_laps.columns
            assert 'LapNumber' in fastest_laps.columns
            assert 'LapTime' in fastest_laps.columns
            assert fastest_laps['Year'].iloc[0] == 2023
            assert fastest_laps['GrandPrix'].iloc[0] == 'TestGP'
            assert fastest_laps['Session'].iloc[0] == 'R'
            assert len(fastest_laps) == 2
    
    def test_get_tyre_strategy(self, client, mock_session, mock_cache_manager):
        """Test getting tyre strategy."""
        with patch.object(client, 'get_session', return_value=mock_session):
            # Test cache miss
            tyre_strategy = client.get_tyre_strategy(2023, 'TestGP')
            
            assert 'Year' in tyre_strategy.columns
            assert 'GrandPrix' in tyre_strategy.columns
            assert 'Driver' in tyre_strategy.columns
            assert 'Stint' in tyre_strategy.columns
            assert 'Compound_first' in tyre_strategy.columns
            assert 'LapNumber_min' in tyre_strategy.columns
            assert 'LapNumber_max' in tyre_strategy.columns
            assert tyre_strategy['Year'].iloc[0] == 2023
            assert tyre_strategy['GrandPrix'].iloc[0] == 'TestGP'
            
            # Test cache hit
            mock_cache_manager.reset_mock()
            mock_cache_manager.get.return_value = tyre_strategy
            
            result = client.get_tyre_strategy(2023, 'TestGP')
            assert result.equals(tyre_strategy)
            mock_cache_manager.get.assert_called_once_with('tyre_strategy_2023_TestGP')


if __name__ == "__main__":
    """
    Execute tests when file is run directly.
    """
    import pytest
    import sys
    
    # Run tests in this file with verbosity
    result = pytest.main(["-v", __file__])
    
    # Exit with the same status code pytest would
    sys.exit(result)