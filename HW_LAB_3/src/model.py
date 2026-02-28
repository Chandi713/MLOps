from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pickle
import os

os.makedirs("models", exist_ok=True)


class Polynomial_Regression:
    def __init__(self, degree=2):
        self.degree = degree
        self.model_path = os.path.join("models", f"polynomial_regression_degree_{degree}.pkl")

        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            self.status = "loaded"
        
        else:
            effective_degree = min(self.degree, 3)
            self.model = Pipeline(
                [
                    ("scaler", MinMaxScaler(feature_range=(0, 1))),
                    ("polyfeatures", PolynomialFeatures(degree=effective_degree)),
                    ("reg", LinearRegression()),
                ]
            )
            self.status = "new"

    def reshape(self, X):
        X = X.reshape(-1, 1)
        return X

    def fit(self, x, y):
        x = self.reshape(x)
        y = y.ravel()
        self.model.fit(x, y)
        with open(self.model_path, "wb") as file:
            pickle.dump(self.model, file)
        self.status = "trained"

    def predict(self, x):
        x = self.reshape(x)
        return self.model.predict(x)