import numpy as np
from devito import Eq, Operator, Function, TimeFunction, solve
from examples.seismic import PointSource, Receiver


def FD_kernel(model, u, space_order):
    """
    VTI finite difference kernel. The equation solved is:

    m * u.dt2 =  t1 * (u.dx2 + u.dy2) + t2 * u.dz2 - damp * u.dt

    where t1 and t2 are defined as:

    t1 = [(1+2 *epsilon) + sn]
    t2 = (1 + sn)

    and

    sn is a fraction with numerator -2(epsilon-delta)*(u.dxc**2 + u.dyc**2)*u.dzc**2
    and denominator
    u.dzc**4 + 2*(1+epsilon)*(u.dxc**2 + u.dyc**2)*u.dzc**2 +
    (1+2*epsilon)*(u.dxc**2 + u.dyc**2)**2

    Epsilon and delta are the Thomsen parameters.

    References:
        * Oscar Mojica, Reynam Pestana, Lucas Bitencourt and Átila Saraiva. "Space-domain
          pure qP-wave equations for modeling in 2D TI media." SEG Technical Program
          Expanded Abstracts 2022. Society of Exploration Geophysicists, 2022.

    Parameters
    ----------
    u : TimeFunction
        VTI field.
    space_order : int
        Space discretization order.

    Returns
    -------
    the stencil corresponding to the second order VTI wave equation.
    """
    # Thomsem parameters setup
    m, damp = model.m, model.damp
    delta, epsilon = model.delta, model.epsilon
    eps = np.finfo(model.dtype).eps

    if len(model.shape) == 2:
        dur_p2 = u.dxc**2
        dur_p4 = u.dxc**4
        duz_p2 = u.dyc**2
        duz_p4 = u.dyc**4
        ddur = u.dx2
        dduz = u.dy2
    else:
        dur_p2 = (u.dxc**2 + u.dyc**2)
        dur_p4 = (u.dxc**2 + u.dyc**2)**2
        duz_p2 = u.dzc**2
        duz_p4 = u.dzc**4
        ddur = u.dx2 + u.dy2
        dduz = u.dz2

    numerator = (-2.*(epsilon-delta)*dur_p2*duz_p2)
    denominator = (duz_p4 + (2. + 2.*epsilon)*dur_p2*duz_p2 +
                   (1. + 2*epsilon)*dur_p4 + eps)
    sn = numerator/denominator
    term1 = (1. + 2.*epsilon + sn)*ddur
    term2 = (1. + sn)*dduz
    pde = m*u.dt2 - (term1 + term2) + damp * u.dt

    # Stencil
    stencil = Eq(u.forward, solve(pde, u.forward))

    return stencil


def ForwardOperator(model, geometry, space_order=4,
                    save=False, **kwargs):
    """
    Construct an forward modelling operator in an vti media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer, optional
        Saving flag, True saves all time steps. False saves three timesteps.
        Defaults to False.
    """

    dt = model.grid.time_dim.spacing
    m = model.m
    time_order = 2

    # Create symbols for forward wavefield, source and receivers
    u = TimeFunction(name='u', grid=model.grid, staggered=None,
                     save=geometry.nt if save else None,
                     time_order=time_order, space_order=space_order)
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    # FD kernels of the PDE
    stencil = FD_kernel(model, u, space_order)

    # Source and receivers
    expr = src * dt**2 / m
    stencil += src.inject(field=u.forward, expr=expr)
    stencil += rec.interpolate(expr=u)

    # Substitute spacing terms to reduce flops
    return Operator(stencil, subs=model.spacing_map, name='ForwardVTI', **kwargs)
