class Function:
    def __init__(self, wrapped, vjps):
        self.wrapped = wrapped
        self.vjps = vjps

    def __call__(self, *args, **kwargs):
        argvals = list(args)

        parents = []
        for i, arg in enumerate(args):
            if isinstance(arg, Node):
                argvals[i] = arg.value
                parents.append((i, arg))

        result_value = self.wrapped(*argvals, **kwargs)
        if parents:
            return Node(value=result_value, recipe=(self, argvals, kwargs, parents))
        else:
            return result_value

    def get_vjp(self, argnum):
        return self.vjps[argnum]


class Node:
    def __init__(self, value, recipe):
        self.value = value
        self.recipe = recipe

    def parents(self):
        return [p for i, p in self.recipe[-1]]

    @classmethod
    def new_root(cls, value):
        return cls(value, (None, (), {}, []))


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
