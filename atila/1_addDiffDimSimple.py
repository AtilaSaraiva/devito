import numpy as np

a = np.ones((2, 2))

b = np.ones((5, 5))


def diffShapeSum(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            b[i, j] = b[i, j] + a[i, j]

    return b


print(diffShapeSum(a, b))
