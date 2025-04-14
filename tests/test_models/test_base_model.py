"""
Tests for the base model of the F1 prediction project.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import datetime
from models.base_model import F1PredictionModel

# Create a concrete implementation of the abstract base class for testing
class TestModel(F1PredictionModel):
    """Concrete implementation of F1PredictionModel for testing."""
    
    def __init__(self):
        super().__init__(name='test_model', version='0.1')
        self.features = ['feature1', 'feature2']
        self.target = 'Position'
        
    def prepare_features(self, df):
        """Implement abstract method."""
        return df[self.features]
        
    def train(self, X, y):
        """Implement abstract method."""
        self.is_trained = True
        self.training_date = datetime.now()
        # Simple model that predicts the average
        self.model = {'mean': y.mean()}
        
    def predict(self, X):
        """Implement abstract method."""
        if not self.is_trained:
            return np.array([])
        # Return the mean for all samples
        return np.full(len(X), self.model['mean'])


class TestBaseModel(unittest.TestCase):
    """Test cases for the F1PredictionModel base class."""
    
    def setUp(self):
        """Set up test data and model instance."""
        self.model = TestModel()
        
        # Create sample data
        self.X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'PER']
        })
        
        self.y = pd.Series([1, 2, 3, 4, 5], name='Position')
        
        # Create sample test data
        self.X_test = pd.DataFrame({
            'feature1': [2, 3, 4],
            'feature2': [4, 3, 2],
            'Driver': ['NOR', 'ALO', 'RUS']
        })
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, 'test_model')
        self.assertEqual(self.model.version, '0.1')
        self.assertFalse(self.model.is_trained)
        self.assertIsNone(self.model.training_date)
        self.assertEqual(len(self.model.metrics), 0)
    
    def test_train_method(self):
        """Test the training process."""
        self.model.train(self.X, self.y)
        
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.training_date)
        self.assertEqual(self.model.model['mean'], 3)  # Mean of [1,2,3,4,5]
    
    def test_predict_method(self):
        """Test the prediction process."""
        # Model should not predict when not trained
        empty_pred = self.model.predict(self.X_test)
        self.assertEqual(len(empty_pred), 0)
        
        # Train the model
        self.model.train(self.X, self.y)
        
        # Test predictions
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), 3)
        
        # All predictions should be the mean (3)
        self.assertTrue((predictions == 3).all())
    
    def test_evaluate_method(self):
        """Test the evaluation process."""
        # Train the model
        self.model.train(self.X, self.y)
        
        # Evaluate the model
        metrics = self.model.evaluate(self.X, self.y)
        
        # Check that metrics were computed
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # Mean Absolute Error should be the average distance from predictions (all 3) to actual values
        expected_mae = np.mean([abs(3-1), abs(3-2), abs(3-3), abs(3-4), abs(3-5)])
        self.assertAlmostEqual(metrics['mae'], expected_mae, places=4)
    
    def test_save_load_methods(self):
        """Test saving and loading the model."""
        # Train the model
        self.model.train(self.X, self.y)
        
        # Create a temporary file for saving
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp:
            temp_path = temp.name
        
        try:
            # Save the model
            save_result = self.model.save(temp_path)
            self.assertTrue(save_result)
            
            # Create a new model for loading
            new_model = TestModel()
            self.assertFalse(new_model.is_trained)
            
            # Load the model
            load_result = new_model.load(temp_path)
            self.assertTrue(load_result)
            
            # Check that model states match
            self.assertEqual(new_model.name, self.model.name)
            self.assertEqual(new_model.version, self.model.version)
            self.assertTrue(new_model.is_trained)
            self.assertEqual(new_model.model['mean'], self.model.model['mean'])
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_predict_race_results(self):
        """Test prediction of race results."""
        # Train the model
        self.model.train(self.X, self.y)
        
        # Predict race results
        results = self.model.predict_race_results(self.X)
        
        # Check result format
        self.assertEqual(len(results), 5)
        self.assertIn('Driver', results.columns)
        self.assertIn('PredictedPosition', results.columns)
        
        # Check that the drivers are preserved
        for driver in self.X['Driver']:
            self.assertIn(driver, results['Driver'].values)
    
    def test_string_representation(self):
        """Test string and representation methods."""
        str_repr = str(self.model)
        self.assertIn('test_model', str_repr)
        self.assertIn('0.1', str_repr)
        
        detailed_repr = repr(self.model)
        self.assertIn('test_model', detailed_repr)
        self.assertIn('0.1', detailed_repr)
        self.assertIn('Untrained', detailed_repr)


if __name__ == '__main__':
    unittest.main()