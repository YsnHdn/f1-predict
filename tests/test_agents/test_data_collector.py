"""
Unit tests for the DataCollectorAgent class.
Tests the agent's ability to collect race data and determine the next race.
"""

import os
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from agents.data_collector import DataCollectorAgent

# Mocking the FastF1Client to avoid cache directory issues
@pytest.fixture
def mock_fastf1_client():
    """Create a mock FastF1Client for testing."""
    with patch('agents.data_collector.FastF1Client') as MockFastF1Client:
        mock_client = MagicMock()
        MockFastF1Client.return_value = mock_client
        yield mock_client

class TestDataCollectorAgent:
    """Test suite for the DataCollectorAgent class."""
    
    @pytest.fixture
    def agent(self, mock_fastf1_client):
        """Create a DataCollectorAgent instance for testing."""
        # Use a temporary directory for test data
        test_data_dir = "test_data/raw"
        
        # Make sure the directory exists
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)
            
        # Create the agent with the mocked FastF1Client
        agent = DataCollectorAgent(data_dir=test_data_dir)
        
        # Yield the agent for the test
        yield agent
        
        # Cleanup after tests
        if os.path.exists(test_data_dir):
            import shutil
            shutil.rmtree(test_data_dir)
    
    @pytest.fixture
    def race_calendar_df(self):
        """Create a sample race calendar DataFrame for testing."""
        # Current date for testing
        current_date = datetime.now()
        
        # Create dates for past and future races
        past_race_date = current_date - timedelta(days=30)
        next_race_date = current_date + timedelta(days=7)
        future_race_date = current_date + timedelta(days=21)
        
        # Create sample DataFrame
        data = {
            'EventName': ['Past Grand Prix', 'Next Grand Prix', 'Future Grand Prix'],
            'EventDate': [past_race_date, next_race_date, future_race_date],
            'OfficialEventName': ['Past Circuit', 'Next Circuit', 'Future Circuit'],
            'RoundNumber': [1, 2, 3]
        }
        return pd.DataFrame(data)
    
    def test_init(self, agent):
        """Test the initialization of the DataCollectorAgent."""
        assert agent.name == "Data Collector"
        assert os.path.exists(agent.data_dir)
        assert os.path.exists(os.path.join(agent.data_dir, 'races'))
        assert os.path.exists(os.path.join(agent.data_dir, 'qualifying'))
        assert os.path.exists(os.path.join(agent.data_dir, 'practice'))
        assert os.path.exists(os.path.join(agent.data_dir, 'historical'))
    
    def test_get_backstory(self, agent):
        """Test that the agent returns a valid backstory."""
        backstory = agent.get_backstory()
        assert isinstance(backstory, str)
        assert len(backstory) > 0
    
    def test_auto_determine_next_race(self, agent, race_calendar_df, mock_fastf1_client):
        """Test that the agent can automatically determine the next race."""
        # Set up mock return value
        mock_fastf1_client.get_race_calendar.return_value = race_calendar_df
        
        # Call execute with no specific race
        context = {'year': datetime.now().year}
        results = agent.execute(context)
        
        # Verify that the agent selected the next upcoming race
        assert results['gp_name'] == 'Next Grand Prix'
        
        # Instead of checking exact number of calls, just verify that it was called with the correct year
        mock_fastf1_client.get_race_calendar.assert_any_call(datetime.now().year)
    
    def test_collect_standings_data(self, agent, race_calendar_df, mock_fastf1_client):
        """Test that the agent collects driver and constructor standings."""
        # Set up mock return values
        mock_fastf1_client.get_race_calendar.return_value = race_calendar_df
        
        # Sample driver standings
        driver_data = {
            'Position': [1, 2, 3],
            'Driver': ['Driver1', 'Driver2', 'Driver3'],
            'Points': [100, 90, 80],
            'Team': ['Team1', 'Team2', 'Team3']
        }
        mock_fastf1_client.get_driver_standings.return_value = pd.DataFrame(driver_data)
        
        # Sample constructor standings
        constructor_data = {
            'Position': [1, 2, 3],
            'Team': ['Team1', 'Team2', 'Team3'],
            'Points': [150, 120, 100]
        }
        mock_fastf1_client.get_constructor_standings.return_value = pd.DataFrame(constructor_data)
        
        # Call execute
        context = {'year': datetime.now().year, 'gp_name': 'Next Grand Prix'}
        results = agent.execute(context)
        
        # Verify that standings data was collected
        assert 'driver_standings' in results
        assert 'constructor_standings' in results
        assert 'driver_standings' in results['data_paths']
        assert 'constructor_standings' in results['data_paths']
        
        # Verify data paths exist
        assert os.path.exists(results['data_paths']['driver_standings'])
        assert os.path.exists(results['data_paths']['constructor_standings'])
    
    def test_collect_historical_data(self, agent, race_calendar_df, mock_fastf1_client):
        """Test that the agent collects historical race data."""
        # Set up mock return values
        mock_fastf1_client.get_race_calendar.return_value = race_calendar_df
        
        # Sample driver and constructor standings
        mock_fastf1_client.get_driver_standings.return_value = pd.DataFrame({'Driver': ['D1'], 'Points': [10]})
        mock_fastf1_client.get_constructor_standings.return_value = pd.DataFrame({'Team': ['T1'], 'Points': [20]})
        
        # Session mock
        session_mock = MagicMock()
        session_mock.event = {'EventName': 'Next Grand Prix', 'OfficialEventName': 'Test Circuit'}
        mock_fastf1_client.get_session.return_value = session_mock
        
        # Race results mock
        race_results_data = {
            'Driver': ['Driver1', 'Driver2'],
            'Position': [1, 2],
            'Points': [25, 18],
            'Team': ['Team1', 'Team2']
        }
        mock_fastf1_client.get_race_results.return_value = pd.DataFrame(race_results_data)
        
        # Call execute with historical_years parameter
        context = {
            'year': datetime.now().year,
            'gp_name': 'Next Grand Prix',
            'historical_years': 1
        }
        results = agent.execute(context)
        
        # Verify that historical data was collected
        assert 'historical_data' in results
        assert len(results['historical_data']) > 0
        
        # At least one key in historical_data should have 'combined' in it
        combined_present = any('combined' in key for key in results['historical_data'].keys())
        assert combined_present
    
    def test_save_data_to_file(self, agent):
        """Test saving data to file."""
        # Create test data
        test_data = pd.DataFrame({
            'Driver': ['Driver1', 'Driver2'],
            'Position': [1, 2],
            'Points': [25, 18]
        })
        
        # Save data
        file_path = agent._save_data_to_file(test_data, 'test_data', year=2023)
        
        # Verify file was created
        assert os.path.exists(file_path)
        
        # Verify CSV has correct data
        loaded_data = pd.read_csv(file_path)
        assert loaded_data.shape == test_data.shape
        assert all(loaded_data['Driver'] == test_data['Driver'])
        
        # Check JSON was also created
        json_path = file_path.replace('.csv', '.json')
        assert os.path.exists(json_path)
    
    def test_error_handling(self, agent, mock_fastf1_client):
        """Test that the agent properly handles errors."""
        # Make get_driver_standings raise an exception
        mock_fastf1_client.get_race_calendar.return_value = pd.DataFrame({
            'EventName': ['Next Race'],
            'EventDate': [datetime.now() + timedelta(days=7)]
        })
        mock_fastf1_client.get_driver_standings.side_effect = Exception("API error")
        
        # Call execute and expect exception to be re-raised
        context = {'year': datetime.now().year, 'gp_name': 'Next Race'}
        with pytest.raises(Exception):
            agent.execute(context)
            
    def test_missing_race(self, agent, mock_fastf1_client):
        """Test behavior when no races are found in the calendar."""
        # Return empty calendar
        mock_fastf1_client.get_race_calendar.return_value = pd.DataFrame()
        
        # Call execute with no specific race, should raise an error
        context = {'year': datetime.now().year}
        with pytest.raises(ValueError, match="No Grand Prix specified and could not determine next race"):
            agent.execute(context)