import numpy as np


def declareSubdomain(shape, leftTop, grid):
    return {'shape': shape, 'leftTop': leftTop, 'grid': grid}


grid = (5, 5)
subdomain = declareSubdomain(shape=(2, 2), leftTop=(1, 1), grid=grid)

a = np.ones(subdomain['shape'])

b = np.ones(grid)


def diffShapeSum(a, b, subdomain):
    y_offset, x_offset = subdomain['leftTop']
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            b[i+y_offset, j+x_offset] = b[i+y_offset, j+x_offset] + a[i, j]

    return b


print(diffShapeSum(a, b, subdomain))
