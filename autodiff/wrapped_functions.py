import numpy as np

from autodiff.tracing import Function


add = Function(wrapped=np.add, vjps=[lambda g, ans, x, y: g, lambda g, ans, x, y: g])


def subtract_vjp1(g, ans, x, y):
    return g

def subtract_vjp2(g, ans, x, y):
    return -g

subtract = Function(wrapped=np.subtract, vjps=[subtract_vjp1, subtract_vjp2])

# TODO: add Function objects wrapping the necessary numpy functions for linear regression (multiply, matvecmul, mean)
