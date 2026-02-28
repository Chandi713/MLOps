import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Form, status, HTTPException
from fastapi.responses import HTMLResponse
from data import generate_data
from train import train_model
from predict import evaluate_model

app = FastAPI()

# -------------------------
# Main Page (GET)
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <head>
            <title>Polynomial Regression Predictor</title>
        </head>
        <body>
            <h2>Polynomial Regression</h2>
            <form action="/predict" method="post">
                <label>Number of Points:</label><br>
                <input type="number" name="points" min="10" max="1000" required><br><br>

                <label>Polynomial Degree:</label><br>
                <input type="number" name="degree" min="1" max="15" required><br><br>

                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

# -------------------------
# Predict (POST)
# -------------------------

@app.post("/predict", response_class=HTMLResponse)
async def predict(points: int = Form(...), degree: int = Form(...)):
    try:
        x_train, y_train, x_test, y_test = generate_data(points)
        model = train_model(x_train, y_train, degree)
        metrics = evaluate_model(model, x_train, y_train, x_test, y_test)

        return f"""
        <html>
            <head>
                <title>Prediction Results</title>
            </head>
            <body>
                <h2>Results</h2>
                <p><strong>Points:</strong> {points}</p>
                <p><strong>Degree:</strong> {degree}</p>

                <h3>Training Metrics</h3>
                <p>MSE: {metrics['train_mse']:.4f}</p>
                <p>R²: {metrics['train_r2']:.4f}</p>

                <h3>Testing Metrics</h3>
                <p>MSE: {metrics['test_mse']:.4f}</p>
                <p>R²: {metrics['test_r2']:.4f}</p>

                <br>
                <a href="/">Run Again</a>
            </body>
        </html>
        """

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
