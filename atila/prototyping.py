from devito import Grid, Function, Eq, Operator
from devito import SubDomain

class Middle(SubDomain):
    name = 'middle'

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 3, 4), y: ('middle', 4, 3)}


mid = Middle()

grid = Grid(shape=(10, 10), subdomains=(mid, ))

f = Function(name='f', grid=grid.subdomains['middle'])

print(f.data.shape)

# eq = Eq(f, f+1, subdomain=grid.subdomains['middle'])

# op = Operator(eq)()

# print(f.data)

# print(type(grid.subdomains['middle']))

# print(grid.subdomains['middle'].dimensions)
# print(grid.subdomains['middle'].shape)
# print(type(grid))
