import numpy as np

from sol_autodiff.tracing import Node, backward


def _substitute(tup, idx, value):
    ret = list(tup)
    ret[idx] = value
    return tuple(ret)


def grad(fun, argnum=0):
    def grad_fun(*args, **kwargs):
        start_node = Node.new_root(args[argnum])
        unary_fun = lambda x: fun(*_substitute(args, argnum, start_node), **kwargs)
        end_node = unary_fun(start_node)
        if end_node is None:
            return np.zeros_like(start_node.value)
        else:
            return backward(np.ones_like(end_node.value), end_node)
    return grad_fun
