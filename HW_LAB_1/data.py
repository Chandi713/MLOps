import numpy as np
import matplotlib.pyplot as plt

def generate_data(points=80):
    # 1. Generate random x-coordinates
    np.random.seed(42)
    num_points = points
    x = np.random.uniform(0, 4 * np.pi, num_points)
    
    # 2. Compute y-coordinates (sin wave)
    y_pure = np.sin(x)
    
    # 3. Add Gaussian noise to y-coordinates
    noise_mean = 0
    noise_std = 0.2
    noise = np.random.normal(noise_mean, noise_std, num_points)
    y_noisy = y_pure + noise

    x_train = x
    y_train = y_noisy
    x_test = np.random.uniform(0, 4 * np.pi, 100)
    y_test = np.sin(x_test)
    return x_train, y_train, x_test, y_test