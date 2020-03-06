import numpy as np

from util import generate_training_examples
from autodiff.grad import grad
from autodiff.wrapped_functions import multiply, subtract, matvecmul, mean
from sol_linear_regression import numerical_grad


def l2_loss(weights, x, y_gold):
    y_pred = matvecmul(x, weights)
    diff = subtract(y_pred, y_gold)
    loss = multiply(diff, diff)
    return mean(loss)


if __name__ == '__main__':
    coeffs = [1, -1, 7, 0, 0.5, 10]
    num_examples = 10
    x, y_gold = generate_training_examples(num_examples, coeffs, 0.0)

    weights = np.zeros_like(coeffs)

    print("=" * 20)
    print("gradient check...")
    loss_grad_fn = grad(l2_loss)
    gg = loss_grad_fn(weights, x, y_gold)
    ng = numerical_grad(l2_loss, weights, x, y_gold, dw=1e-6)
    mean_rel_diff = 2.0 * np.mean(np.abs(gg - ng) / np.abs(gg + ng))
    print(f"rel_diff = {mean_rel_diff}")

    print("=" * 20)
    step_size = 0.01
    num_steps = 5000
    num_print = 100
    for i in range(num_steps):
        weights -= step_size * loss_grad_fn(weights, x, y_gold)
        if i % num_print == 0:
            print(f"i={i:d}, loss={l2_loss(weights, x, y_gold):.6e}")

    print("=" * 20)
    print(f"loss after {num_steps} steps is {l2_loss(weights, x, y_gold):.3f}")
