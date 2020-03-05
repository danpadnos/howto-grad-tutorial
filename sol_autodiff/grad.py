import numpy as np

from sol_autodiff.tracing import Node


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


def backward(g, end_node):
    outgrads = {end_node: g}
    for node in toposort(end_node):
        outgrad = outgrads.pop(node)
        fun, args, kwargs, parents = node.recipe
        for argnum, parent in parents:
            vjp = fun.get_vjp(argnum)
            parent_grad = vjp(outgrad, node.value, *args, **kwargs)
            outgrads[parent] = outgrads.get(parent, 0) + parent_grad
    return outgrad


def toposort(end_node):
    child_counts = {}
    stack = [end_node]
    while stack:
        node = stack.pop()
        if node in child_counts:
            child_counts[node] += 1
        else:
            child_counts[node] = 1
            stack.extend(node.parents())

    childless_nodes = [end_node]
    while childless_nodes:
        node = childless_nodes.pop()
        yield node
        for parent in node.parents():
            if child_counts[parent] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[parent] -= 1