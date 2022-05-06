import numpy as np
from devito import Eq, Operator, Function, TimeFunction, solve, cos, sin
from examples.seismic import PointSource, Receiver

def FD_kernel(model, u, space_order):
    """
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

    # Get symbols from model
    theta = model.theta
    delta = model.delta
    epsilon = model.epsilon
    m = model.m
    # Use trigonometric functions from Devito
    costheta  = cos(theta)
    sintheta  = sin(theta)
    cos2theta = cos(2*theta)
    sin2theta = sin(2*theta)
    sin4theta = sin(4*theta)

    if len(model.shape) == 2:
        # Main equations
        t1 = 2*epsilon*costheta**4 + 2*delta*sintheta**2*costheta**2
        t2 = 2*epsilon*sintheta**4 + 2*delta*sintheta**2*costheta**2
        t3 = -4*epsilon*sin2theta*costheta**2 + delta*sin4theta
        t4 = -4*epsilon*sin2theta*sintheta**2 - delta*sin4theta
        t5 = 3*epsilon*sin2theta**2 - delta*sin2theta**2 + 2*delta*cos2theta**2

        b1 = (1+t1)
        b2 = (1-t1+t5)
        b3 = (t1+t2-t5)
        b4 = t3
        b5 = (t4-t3)

        s = 1./(1 + (u.dxc**2)/(u.dyc**2+eps))

        sn = (b1*u.dx2 + (b2 + b3*s)*u.dy2 + (b4 + b5*s)*u.dx.dy)
        pde = m*u.dt2 - sn + damp * u.dt

    else:
        phi = model.phi

        cosphi  = cos(phi)
        sinphi  = sin(phi)
        cos2phi = cos(2*phi)
        sin2phi = sin(2*phi)

        s_xy = ( u.dxc**2 + u.dyc**2 ) / (u.dzc**2)
        s_xz = ( u.dxc**2 + u.dzc**2 ) / (u.dyc**2)
        s_yz = ( u.dyc**2 + u.dzc**2 ) / (u.dxc**2)

        st_xy = 1 / ( 1 + s_xy)
        st_xz = 1 / ( 1 + s_xz)
        st_yz = 1 / ( 1 + s_yz)

        c1 = 1 + 2*epsilon
        c2 = 2 * (delta - 2*epsilon)
        c3 = (epsilon - delta)
        t3c3 = 3 * c3
        t4c3 = 4 * c3
        t6c3 = 6 * c3

        A11   = c1 + c2 * sintheta**2 * cosphi**2
        A22   = c1 + c2 * sintheta**2 * sinphi**2
        A33   = c1 + c2 * costheta**2
        A12   = c2 * sintheta**2 * sin2phi
        A13   = c2 * sin2theta   * cosphi
        A23   = c2 * sin2theta   * sinphi
        A1111 = 2*c3 * sintheta**4 * cosphi**4
        A2222 = 2*c3 * sintheta**4 * sinphi**4
        A3333 = 2*c3 * costheta**4
        A1112 = t4c3 * sintheta**4 * sin2phi * cosphi**2
        A1113 = t4c3 * sin2theta * sintheta**2 * cosphi**3
        A1222 = t4c3 * sintheta**4 * sin2phi * sinphi**2
        A2223 = t4c3 * sin2theta * sintheta**2 * sinphi**3
        A1333 = t4c3 * sin2theta * costheta**2 * cosphi
        A2333 = t4c3 * sin2theta * costheta**2 * sinphi
        A1122 = t3c3 * sintheta**4 * sin2phi**2
        A1133 = t3c3 * sin2theta**2 * cosphi**2
        A2233 = t3c3 * sin2theta**2 * sinphi**2
        A1123 = t6c3 * sin2theta * sintheta**2 * sin2phi * cosphi
        A1223 = t6c3 * sin2theta * sintheta**2 * sin2phi * sinphi
        A1233 = t3c3 * sin2theta**2 * sin2phi

        At11   = A11 + (A1133 + A1122 - A2233)/2
        At22   = A22 + (-A1133 + A1122 + A2233)/2
        At33   = A33 + (A1133 - A1122 + A2233)/2
        At12   = A12 + A1233
        At13   = A13 + A1223
        At23   = A23 + A1123
        At1111 = A1111 + (-A1133 - A1122 + A2233)/2
        At2222 = A2222 + (A1133 - A1122 - A2233)/2
        At3333 = A3333 + (-A1133 + A1122 - A2233)/2
        At1112 = A1112 - A1233
        At1113 = A1113 - A1223
        At1222 = A1222 - A1233
        At2223 = A2223 - A1123
        At1333 = A1333 - A1223
        At2333 = A2333 - A1123

        t1 = At11 + At1111 * st_yz
        t2 = At22 + At2222 * st_xz
        t3 = At33 + At3333 * st_xy
        t4 = At12 + At1112 * st_yz + At1222 * st_xz
        t5 = At13 + At1113 * st_yz + At1333 * st_xy
        t6 = At23 + At2223 * st_xz + At2333 * st_xy

        H = t1*u.dx2 + t2*u.dy2 + t3*u.dz2 + t4*u.dx.dy + t5*u.dx.dz + t6*u.dy.dz

        pde = m*u.dt2 + H + damp * u.dt

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
    return Operator(stencil, subs=model.spacing_map, name='ForwardTTI', **kwargs)
