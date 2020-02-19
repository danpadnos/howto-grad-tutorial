import numpy as np

from util import generate_training_examples


def numerical_grad(loss_fn, weights, x, y_gold, dw):
    loss = loss_fn(weights, x, y_gold)

    grad = np.zeros_like(weights, dtype=np.float)
    for i in range(len(weights)):
        weights[i] += dw
        dloss = loss_fn(weights, x, y_gold) - loss
        weights[i] -= dw
        grad[i] = dloss / dw

    return grad


def l2_loss(weights, x, y_gold):
    y_pred = predict(weights, x)
    diff = np.subtract(y_pred, y_gold)
    loss = np.multiply(diff, diff)
    return np.mean(loss, axis=0)


def loss_grad(weights, x, y_gold):
    y_pred = predict(weights, x)
    diff = y_pred - y_gold
    coeff_grad = np.mean(2 * diff[..., np.newaxis] * x, axis=0)
    return coeff_grad


def predict(weights, x):
    return np.matmul(x, weights)


if __name__ == '__main__':
    coeffs = [1, -1, 7, 0, 0.5, 10]
    num_examples = 10
    x, y_gold = generate_training_examples(num_examples, coeffs, 0.0)

    weights = np.zeros_like(coeffs)

    print("=" * 20)
    print("gradient check...")
    gg = loss_grad(weights, x, y_gold)
    ng = numerical_grad(l2_loss, weights, x, y_gold, dw=1e-6)
    mean_rel_diff = 2.0 * np.mean(np.abs(gg - ng) / np.abs(gg + ng))
    print(f"rel_diff = {mean_rel_diff}")

    print("=" * 20)
    step_size = 0.01
    num_steps = 5000
    num_print = 100
    for i in range(num_steps):
        weights -= step_size * loss_grad(weights, x, y_gold)
        if i % num_print == 0:
            print(f"i={i:d}, loss={l2_loss(weights,x,y_gold):.6e}")

    print("=" * 20)
    print(f"loss after {num_steps} steps is {l2_loss(weights,x,y_gold):.3f}")
