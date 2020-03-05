import numpy as np

from sol_autodiff.tracing import Function


add = Function(wrapped=np.add, vjps=[lambda g, ans, x, y: g, lambda g, ans, x, y: g])


def subtract_vjp1(g, ans, x, y):
    return g

def subtract_vjp2(g, ans, x, y):
    return -g

subtract = Function(wrapped=np.subtract, vjps=[subtract_vjp1, subtract_vjp2])


def multiply_vjp1(g, ans, x, y):
    return g * y

def multiply_vjp2(g, ans, x, y):
    return g * x

multiply = Function(wrapped=np.multiply, vjps=[multiply_vjp1, multiply_vjp2])


def divide_vjp1(g, ans, x, y):
    return g / y

def divide_vjp2(g, ans, x, y):
    return -g * x / y**2

divide = Function(wrapped=np.divide, vjps=[divide_vjp1, divide_vjp2])


def mean_vjp(g, ans, x):
    return g * np.ones_like(x) / len(x)

mean = Function(wrapped=np.mean, vjps=[mean_vjp])


def matvecmul_vjp1(g, ans, x, y):
    return np.einsum('i,j->ij', g, y)  #np.outer(g, y)  #g[:, None] * get_value(y)

def matvecmul_vjp2(g, ans, x, y):
    return np.einsum('i,ij->j', g, x)  # np.dot(g, get_value(x))

matvecmul = Function(wrapped=np.matmul, vjps=[matvecmul_vjp1, matvecmul_vjp2])