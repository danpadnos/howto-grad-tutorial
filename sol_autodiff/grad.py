import numpy as np

from sol_autodiff.tracing import Node


def _substitute(tup, idx, value):
    ret = list(tup)
    ret[idx] = value
    return tuple(ret)


def grad(fun, argnum=0):
    def grad_fun(*args, **kwargs):
        start_node = Node.make_root(args[argnum])
        end_node = fun(*_substitute(args, argnum, start_node), **kwargs)
        if end_node is None:
            return np.zeros_like(start_node.value)
        else:
            return backward(end_node)
    return grad_fun


def backward(end_node):
    node2grad = {end_node: 1.0}
    for node in toposort(end_node):
        current_grad = node2grad.pop(node)
        fun, args, kwargs = node.recipe
        for argnum, parent in node.parents:
            vjp = fun.get_vjp(argnum)
            parent_grad = vjp(current_grad, node.value, *args, **kwargs)
            node2grad[parent] = node2grad.get(parent, 0.0) + parent_grad
    return current_grad


def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parent_nodes())

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parent_nodes():
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1
