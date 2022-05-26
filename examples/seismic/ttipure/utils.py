import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker
from examples.seismic import AcquisitionGeometry

def setup_rec_coords(model):
    nrecx = model.shape[0]
    recx = np.linspace(model.origin[0], model.domain_size[0], nrecx)

    if model.dim == 1:
        return recx.reshape((nrecx, 1))
    elif model.dim == 2:
        rec_coordinates = np.empty((nrecx, model.dim))
        rec_coordinates[:, 0] = recx
        rec_coordinates[:, -1] = model.origin[-1] + 2 * model.spacing[-1]
        return rec_coordinates
    else:
        nrecy = model.shape[1]
        recy = np.linspace(model.origin[1], model.domain_size[1], nrecy)
        rec_coordinates = np.empty((nrecx*nrecy, model.dim))
        rec_coordinates[:, 0] = np.array([recx[i] for i in range(nrecx)
                                          for j in range(nrecy)])
        rec_coordinates[:, 1] = np.array([recy[j] for i in range(nrecx)
                                          for j in range(nrecy)])
        rec_coordinates[:, -1] = model.origin[-1] + 2 * model.spacing[-1]
        return rec_coordinates

def setup_geometry(model, tn, f0=0.010):
    # Source and receiver geometries
    src_coordinates = np.empty((1, model.dim))
    src_coordinates[0, :] = np.array(model.domain_size) * .5

    rec_coordinates = setup_rec_coords(model)

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=f0)

    return geometry


def plotTimeSlices (model, u, origin=(0.,0.,0.)):
    # Some useful definitions for plotting if nbl is set to any other value than zero
    shape = model.shape
    origin = model.origin
    spacing = model.spacing
    nbl = model.nbl
    nxpad,nypad,nzpad = shape[0] + 2 * nbl, shape[1] + 2 * nbl, shape[2] + 2 * nbl
    shape_pad   = np.array(shape) + 2 * nbl
    origin_pad  = tuple([o - s*nbl for o, s in zip(origin, spacing)])
    extent_pad  = tuple([s*(n-1) for s, n in zip(spacing, shape_pad)])

    # Note: flip sense of second dimension to make the plot positive downwards
    plt_extent = [origin_pad[0], origin_pad[0] + extent_pad[0],
                  origin_pad[1] + extent_pad[1], origin_pad[1]]

    dt = model.critical_dt # We will use the dt for the coupled pseudoacoustic equation

    # Plot the wavefields, each normalized to scaled maximum of last time step
    kt = (u.data.shape[0] - 2) - 1
    amax = np.max(np.abs(u.data[kt,:,:,20]))

    nsnaps = 10
    factor = round(u.data.shape[0]/nsnaps)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for count, ax in enumerate(axes.ravel()):
        snapshot = count *factor
        ax.imshow(np.transpose(u.data[snapshot,:,shape[1]//2,:])/amax, cmap="seismic",
                  vmin=-1, vmax=+1, extent=plt_extent)
        ax.plot(model.domain_size[0]* .5, model.domain_size[1]* .5, \
             'red', linestyle='None', marker='*', markersize=8, label="Source")
        ax.grid()
        ax.tick_params('both', length=2, width=0.5, which='major',labelsize=10)
        ax.set_title("Wavefield at t=%.2fms" % (factor*count*dt),fontsize=10)
    for ax in axes[1, :]:
        ax.set_xlabel("X Coordinate (m)",fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Z Coordinate (m)",fontsize=10)

    plt.show()
