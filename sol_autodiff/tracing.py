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
        return cls(value=value, recipe=(None, (), {}, []))


