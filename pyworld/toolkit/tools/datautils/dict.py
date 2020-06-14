class fdict(dict):

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def __setitem__(self, k, v):
        if k not in self:
            super(fdict, self).__setitem__(k,v)
        else:
            raise KeyError("Key: {0} already exists.".format(k))

    def update(self, *args, **kwargs):
        for k,v in dict(*args, **kwargs).items():
            self[k] = v