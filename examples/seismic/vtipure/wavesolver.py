# coding: utf-8
from devito import Function, TimeFunction, warning
from devito.tools import memoized_meth
from examples.seismic.vtipure.operators import ForwardOperator, AdjointOperator


class PureVtiWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.

    Notes
    -----
    space_order must be even and it is recommended to be a multiple of 4
    """
    def __init__(self, model, geometry, space_order=4,
                 **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        if space_order % 2 != 0:
            raise ValueError("space_order must be even but got %s"
                             % space_order)

        if space_order % 4 != 0:
            warning("It is recommended for space_order to be a multiple of 4" +
                    "but got %s" % space_order)

        self.space_order = space_order

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=False):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               space_order=self.space_order,
                               **self._kwargs)

    def forward(self, src=None, rec=None, u=None, model=None,
                save=False, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        u : TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary.
        """
        time_order = 2
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        if u is None:
            u = TimeFunction(name='u', grid=self.model.grid,
                             save=self.geometry.nt if save else None,
                             time_order=time_order,
                             space_order=self.space_order)

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, summary

    def adjoint(self, rec, srca=None, p=None, model=None,
                save=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.
        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        p : TimeFunction, optional
            The computed wavefield first component.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        time_order = 2

        # Source term is read-only, so re-use the default
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the wavefield if not provided
        if p is None:
            p = TimeFunction(name='p', grid=self.model.grid,
                             time_order=time_order,
                             space_order=self.space_order)

        model = model or self.model
        # Pick vp and Thomsen parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))
        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, p=p,
                                      dt=kwargs.pop('dt', self.dt),
                                      **kwargs)
        return srca, p, summary
