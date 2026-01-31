from model import Polynomial_Regression


def train_model(x, y, degree=2):
    model = Polynomial_Regression(degree)
    model.fit(x, y)
    return model