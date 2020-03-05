from autodiff.tracing import Node


def _substitute(tup, idx, value):
    ret = list(tup)
    ret[idx] = value
    return tuple(ret)


def grad(fun, argnum=0):
    def grad_fun(*args, **kwargs):
        start_node = Node.new_root(args[argnum])
        unary_fun = lambda x: fun(*_substitute(args, argnum, start_node), **kwargs)
        # TODO: trace function execution, call backward pass and return gradient value
    return grad_fun


def backward(g, end_node):
    outgrads = {end_node: g}
    for node in toposort(end_node):
        pass
        # TODO: perform reverse accumulation of gradients and return the final gradient at the start node


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