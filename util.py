import numpy as np


def _generate_features(num_examples, num_features):
    x = np.random.random((num_examples, num_features + 1))
    x[:, -1] = 1.0
    return x


def generate_training_examples(num_examples, coeffs, sigma):
    x = _generate_features(num_examples, len(coeffs) - 1)
    y = np.matmul(x, coeffs) + np.random.normal(0, sigma, num_examples)
    return x, y