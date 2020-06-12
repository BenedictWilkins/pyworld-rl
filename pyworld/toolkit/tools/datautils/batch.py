import numpy as np

def __batch_iterate__(data, batch_size):
    _shape = int(data.shape[0]/batch_size) 
    _max = _shape * batch_size
    _data = data[:_max].reshape(_shape, batch_size, *data.shape[1:])
    for x in _data:
        yield x
    if _max < data.shape[0]:
        yield data[_max:]
    
def batch_iterator(*data, batch_size=128, shuffle=False):
    if shuffle:
        data = __shuffle__(*data)
    if len(data) == 1:
        return __batch_iterate__(data[0], batch_size)
    else:
        return zip(*[__batch_iterate__(d, batch_size) for d in data])

def __shuffle__(*data): #sad that this cant be done in place...
    m = max(len(d) for d in data)
    indx = np.arange(m)
    np.random.shuffle(indx)
    data = [d[indx] for d in data]
    return data


if __name__ == "__main__":
    import numpy as np
    from pyworld.toolkit.tools.debugutils import Time

    x = np.zeros((1000,100,100))
    y = np.ones((1000))

    for bx, by in batch_iterator(x,y):
        print(bx.shape, by.shape)

    for bx in batch_iterator(x):
        print(bx.shape)

    

'''
def batch_iterator(data, batch_size=256):
    m = data.shape[0]
    j = 0
    for i in range(batch_size, m, batch_size):
        yield data[j:i]
        j = i
    yield data[j:]


def batch_iterator2(data, batch_size=256):
    _shape = int(data.shape[0]/batch_size) 
    _max = _shape * batch_size
    _data = data[:_max].reshape(_shape, batch_size, *data.shape[1:])
    for x in _data:
        yield x
    if _max < data.shape[0]:
        yield data[_max:]
''' #there seems to be no difference in performance...?