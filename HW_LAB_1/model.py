from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import pickle
import os

class Polynomial_Regression:
    os.makedirs("models", exist_ok=True)

    def __init__(self, degree=2):
        self.degree = degree
        self.model_path = os.path.join("models", f"polynomial_regression_degree_{degree}.pkl")

        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            self.status = "loaded"
        
        else:
            self.model = Pipeline([('polyfeatures', PolynomialFeatures(degree = self.degree)),
                                    ('reg', LinearRegression())])
            self.status = "new"

    def reshape(self, X):
        X = X.reshape(-1, 1)
        return X

    def fit(self, x, y):
        if self.status == "new":
            x = self.reshape(x)
            y = y.ravel()
            self.model.fit(x, y)
            with open(self.model_path, 'wb') as file:
                pickle.dump(self.model, file)
            self.status = "trained"

    def predict(self, x):
        x = self.reshape(x)
        return self.model.predict(x)