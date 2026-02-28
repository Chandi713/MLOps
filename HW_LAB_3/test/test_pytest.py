import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data import generate_data
from model import Polynomial_Regression
from train import train_model
from predict import evaluate_model
from fastapi.testclient import TestClient
from main import app

def test_generate_data_output_shapes():
    x_train, y_train, x_test, y_test = generate_data()
    assert len(x_train) == 80
    assert len(y_train) == 80
    assert len(x_test) == 100
    assert len(y_test) == 100
    
    x_train, y_train, x_test, y_test = generate_data(points=50)
    assert x_train.shape == (50,)
    assert isinstance(x_train, np.ndarray)


def test_model_fit_and_predict():
    unique_degree = 97
    model_path = os.path.join("models", f"polynomial_regression_degree_{unique_degree}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    
    x = np.linspace(0, 10, 100)
    y = x ** 2
    
    model = Polynomial_Regression(degree=unique_degree)
    model.fit(x, y)
    
    predictions = model.predict(np.array([0, 1, 2, 3]))
    assert len(predictions) == 4
    np.testing.assert_array_almost_equal(predictions, np.array([0, 1, 4, 9]), decimal=0)
    
    if os.path.exists(model_path):
        os.remove(model_path)


def test_train_model_produces_valid_predictions():
    unique_degree = 95
    model_path = os.path.join("models", f"polynomial_regression_degree_{unique_degree}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = 2 * x + 1
    
    model = train_model(x, y, degree=unique_degree)
    prediction = model.predict(np.array([11]))
    assert abs(prediction[0] - 23) < 1
    
    if os.path.exists(model_path):
        os.remove(model_path)


def test_evaluate_model_metric_ranges():
    unique_degree = 96
    model_path = os.path.join("models", f"polynomial_regression_degree_{unique_degree}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)
    
    x_train = np.linspace(0, 10, 100)
    y_train = x_train ** 2
    x_test = np.linspace(0, 10, 50)
    y_test = x_test ** 2
    
    model = train_model(x_train, y_train, degree=unique_degree)
    metrics = evaluate_model(model, x_train, y_train, x_test, y_test)
    
    assert metrics["train_mse"] >= 0 and metrics["train_mse"] < 0.1
    assert metrics["train_r2"] > 0.99
    
    if os.path.exists(model_path):
        os.remove(model_path)

@pytest.fixture
def client():
    return TestClient(app)


def test_main_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<form" in response.text
    assert 'name="points"' in response.text
    assert 'name="degree"' in response.text


def test_predict_endpoint(client):
    response = client.post("/predict", data={"points": 50, "degree": 2})
    assert response.status_code == 200
    assert "Results" in response.text
    assert "MSE" in response.text
    assert "R²" in response.text


def test_predict_endpoint_validation(client):
    assert client.post("/predict", data={"degree": 2}).status_code == 422
    assert client.post("/predict", data={"points": 50}).status_code == 422
    assert client.post("/predict", data={"points": "abc", "degree": 2}).status_code == 422
