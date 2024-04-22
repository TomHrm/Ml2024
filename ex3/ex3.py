from sklearn.linear_model import LinearRegression
import numpy as np

# Aufgabe 2
def ridge_regression(X, y, lam):
    n, d = X.shape
    I = np.eye(d)
    A = np.dot(X.T, X) + lam * n * I
    b = np.dot(X.T, y)
    w = np.linalg.solve(A, b)

    return w

# Aufgabe 3
def generate_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    X_poly = np.ones((n_samples, 1))  # Start with a bias term

    for d in range(1, degree + 1):
        for feature in range(n_features):
            X_poly = np.hstack((X_poly, np.power(X[:, feature], d).reshape(-1, 1)))

    return X_poly

def polynomial_least_squares(X, y, degree):
    X_poly = generate_polynomial_features(X, degree)
    A = np.dot(X_poly.T, X_poly)
    b = np.dot(X_poly.T, y)
    w = np.linalg.solve(A, b)
    return w

def polynomial_regression(X, y, degree, lam):
    X_poly = generate_polynomial_features(X, degree)
    n, d = X_poly.shape
    I = np.eye(d)
    A = np.dot(X_poly.T, X_poly) + lam * n * I
    b = np.dot(X_poly.T, y)
    w = np.linalg.solve(A, b)
    return w

# Aufgabe 4
def k_fold_cross_validation(X, y, k, degree):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = n_samples // np.array([k] * k)
    fold_sizes[:n_samples % k] += 1
    current = 0
    errors = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[0:start], indices[stop:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train the model
        weights = polynomial_regression(X_train, y_train, degree, 0.1)

        # Evaluate the model
        X_test_poly = generate_polynomial_features(X_test, degree)
        y_pred = np.dot(X_test_poly, weights)
        error = np.mean((y_pred - y_test) ** 2)  # Mean squared error
        errors.append(error)

        current = stop

    return np.mean(errors)

if __name__ == '__main__':
    print("Aufgabe 2")
    np.random.seed(0)
    n_samples, n_features = 100, 3
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    y = np.dot(X, true_w) + np.random.randn(n_samples) * 0.5  # adding noise
    model = LinearRegression()
    model.fit(X, y)

    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    lambda_param = 1.0
    ridge_weights = ridge_regression(X, y, lambda_param)
    print("Ridge Regression Coefficients:", ridge_weights)

    # Aufgabe 3
    # Generate some synthetic data
    print("Aufgabe 3")
    np.random.seed(0)
    n_samples = 100
    X = np.random.randn(n_samples, 1)  # Single feature for simplicity
    y = 3 * X[:, 0] ** 2 + 2 * X[:, 0] + 1 + np.random.randn(n_samples) * 0.5  # Quadratic relationship

    # Polynomial degree
    degree = 2

    # Solve polynomial least squares regression
    poly_weights = polynomial_least_squares(X, y, degree)
    print("Polynomial Least Squares Coefficients:", poly_weights)

    # Solve polynomial ridge regression
    lambda_param = 1.0
    poly_ridge_weights = polynomial_regression(X, y, degree, lambda_param)
    print("Polynomial Ridge Regression Coefficients:", poly_ridge_weights)

    # Aufgabe 4
    # Load the training dataset
    print("Aufgabe 4")
    print("Loading datasets...")
    train_data = np.load('dataset_poly_train.npy')

    # Load the test dataset
    test_dataset_path = '/mnt/data/dataset_poly_test.npy'  # Modify as needed for your file path
    test_data = np.load('dataset_poly_test.npy')

    # Split the datasets into features and targets
    X_train = train_data[:, 0]  # Assuming first column is feature
    y_train = train_data[:, 1]  # Assuming second column is target
    X_test = test_data[:, 0]  # Assuming first column is feature
    y_test = test_data[:, 1]  # Assuming second column is target

    # Reshape features to be two-dimensional arrays for compatibility with polynomial features function
    X_train = X_train[:, np.newaxis]
    X_test = X_test[:, np.newaxis]

    # Set the polynomial degree and number of folds
    degree = 6
    k = 5
    lam = 0.1


    # Perform k-fold cross validation on training data
    average_error = k_fold_cross_validation(X_train, y_train, k, degree)
    print("Average cross-validation error on training data:", average_error)

    # Train the model on the full training data and evaluate on the test set
    final_weights = polynomial_regression(X_train, y_train, degree, lam)
    X_test_poly = generate_polynomial_features(X_test, degree)
    y_pred_test = np.dot(X_test_poly, final_weights)
    test_error = np.mean((y_pred_test - y_test) ** 2)
    print("Test set mean squared error:", test_error)