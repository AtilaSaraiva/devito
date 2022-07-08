import numpy as np
from scipy import sparse as sp


def declareSubdomain(shape, leftTop, grid):
    return {'shape': shape, 'leftTop': leftTop, 'grid': grid}


grid = (5, 5)
subdomain = declareSubdomain(shape=(2, 2), leftTop=(1, 1), grid=grid)

row = np.array([1, 2, 1, 2])
col = np.array([1, 1, 2, 2])
data = np.array([1, 1, 1, 1])
a = sp.bsr_array((data, (row, col)), shape=grid, dtype=np.int8)
b = np.ones(grid)

b = b + a

print(b)
