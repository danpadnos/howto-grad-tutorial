class Function:
    def __init__(self, wrapped, vjps):
        self.wrapped = wrapped
        self.vjps = vjps

    def __call__(self, *args, **kwargs):
        pass
        # TODO: unpack values from node args, call wrapped function and pack the return value in a node if necessary

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


