import numpy as np

def generate_data(points=80):
    np.random.seed(42)
    x = np.random.uniform(0, 4 * np.pi, points)
    y_pure = np.sin(x)
    noise = np.random.normal(0, 0.2, points)
    y_noisy = y_pure + noise

    x_train = x
    y_train = y_noisy
    x_test = np.random.uniform(0, 4 * np.pi, 100)
    y_test = np.sin(x_test)
    return x_train, y_train, x_test, y_test
