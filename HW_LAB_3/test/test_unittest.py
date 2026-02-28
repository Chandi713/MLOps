import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data import generate_data
from model import Polynomial_Regression
from train import train_model
from predict import evaluate_model


class TestDataModule(unittest.TestCase):
    def test_generate_data_output_shapes(self):
        x_train, y_train, x_test, y_test = generate_data()
        self.assertEqual(len(x_train), 80)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(x_test), 100)
        self.assertEqual(len(y_test), 100)
        
        x_train, y_train, _, _ = generate_data(points=50)
        self.assertEqual(x_train.shape, (50,))
        self.assertIsInstance(x_train, np.ndarray)

    def test_generate_data_value_ranges(self):
        x_train, y_train, x_test, y_test = generate_data(points=100)
        
        self.assertTrue(np.all(x_train >= 0))
        self.assertTrue(np.all(x_train <= 4 * np.pi))
        self.assertTrue(np.all(x_test >= 0))
        self.assertTrue(np.all(x_test <= 4 * np.pi))
        
        expected_y_test = np.sin(x_test)
        np.testing.assert_array_almost_equal(y_test, expected_y_test)


class TestModelModule(unittest.TestCase):
    def test_model_initialization(self):
        model = Polynomial_Regression()
        self.assertEqual(model.degree, 2)
        self.assertIsNotNone(model.model)
        self.assertIn(model.status, ["new", "loaded"])
        
        model = Polynomial_Regression(degree=5)
        self.assertEqual(model.degree, 5)

    def test_model_fit_and_predict(self):
        unique_degree = 87
        model_path = os.path.join("models", f"polynomial_regression_degree_{unique_degree}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        x = np.linspace(0, 10, 100)
        y = x ** 2
        
        model = Polynomial_Regression(degree=unique_degree)
        model.fit(x, y)
        
        predictions = model.predict(np.array([0, 1, 2, 3]))
        self.assertEqual(len(predictions), 4)
        np.testing.assert_array_almost_equal(predictions, np.array([0, 1, 4, 9]), decimal=0)
        
        if os.path.exists(model_path):
            os.remove(model_path)


class TestTrainModule(unittest.TestCase):
    def test_train_model_returns_correct_type(self):
        x = np.linspace(0, 10, 100)
        y = x ** 2
        
        model = train_model(x, y)
        self.assertIsInstance(model, Polynomial_Regression)
        self.assertEqual(model.degree, 2)
        
        model = train_model(x, y, degree=5)
        self.assertEqual(model.degree, 5)

    def test_train_model_produces_valid_predictions(self):
        unique_degree = 85
        model_path = os.path.join("models", f"polynomial_regression_degree_{unique_degree}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y = 2 * x + 1
        
        model = train_model(x, y, degree=unique_degree)
        prediction = model.predict(np.array([11]))
        self.assertLess(abs(prediction[0] - 23), 1)
        
        if os.path.exists(model_path):
            os.remove(model_path)


class TestPredictModule(unittest.TestCase):
    def test_evaluate_model_returns_correct_metrics(self):
        x_train = np.linspace(0, 10, 100)
        y_train = x_train ** 2
        x_test = np.linspace(0, 10, 50)
        y_test = x_test ** 2
        
        model = train_model(x_train, y_train, degree=2)
        metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("train_mse", metrics)
        self.assertIn("train_r2", metrics)
        self.assertIn("test_mse", metrics)
        self.assertIn("test_r2", metrics)

    def test_evaluate_model_metric_ranges(self):
        unique_degree = 86
        model_path = os.path.join("models", f"polynomial_regression_degree_{unique_degree}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        x_train = np.linspace(0, 10, 100)
        y_train = x_train ** 2
        x_test = np.linspace(0, 10, 50)
        y_test = x_test ** 2
        
        model = train_model(x_train, y_train, degree=unique_degree)
        metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
        
        self.assertGreaterEqual(metrics["train_mse"], 0)
        self.assertLess(metrics["train_mse"], 0.1)
        self.assertGreater(metrics["train_r2"], 0.99)
        
        if os.path.exists(model_path):
            os.remove(model_path)


if __name__ == "__main__":
    unittest.main()
