# Implement least squares regression in Python and run it on a multi-dimensional dataset.
# Run it also on a one-dimensional dataset and visualize the result.
# Please use NumPy for solving this exercise. Other Python libraries like scikit-learn or scipy are disallowed here.

import numpy as np
import matplotlib.pyplot as plt

def calc_mean(x):
    return sum(x) / len(x)

def generate_dataset():
    x = np.random.rand(100)
    y = 3 * x + 4 + np.random.randn(100) * 0.1
    return x, y
def least_squares_regression():
    # least squares regression without regularization (lambda = 0)
    x, y = generate_dataset()
    x_mean = calc_mean(x)
    y_mean = calc_mean(y)

    w_num = 0
    w_denom = 0

    for i in range(len(x)):
        w_num += (y[i] - y_mean) * x[i]
        w_denom += (x[i] - x_mean) * x[i]

    w = w_num / w_denom
    b = y_mean - w * x_mean

    return np.array([w, b])


if __name__ == "__main__":
    w, b = least_squares_regression()
    print(w, b)

    x, y = generate_dataset()
    plt.scatter(x, y)
    plt.plot([0, 1], [b, w + b], c='r')
    plt.show()