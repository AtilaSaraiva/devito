from devito import Grid, Function, Eq, Operator
from devito import SubDomain


class Middle(SubDomain):
    name = 'middle'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('right', 2)}


mid = Middle()

grid = Grid(shape=(10, 10), subdomains=(mid, ))

g = Function(name='f', grid=grid)
f = Function(name='f', grid=grid.subdomains['middle'])

g.data = 0
f.data = 1

eq = Eq(g, f)

op = Operator(eq)

op.apply()

print(g.data)

# result why want with the feature implemented
"""
Data([[0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
      [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]], dtype=float32)
"""

