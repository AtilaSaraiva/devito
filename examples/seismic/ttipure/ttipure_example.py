import numpy as np
import pytest
import argparse

from devito import Function, norm, info
from matplotlib import pyplot as plt

from examples.seismic import demo_model, setup_geometry, seismic_args
from examples.seismic.ttipure import PureTtiWaveSolver


def tti_setup(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
              space_order=4, nbl=10, preset='layers-tti',
              **kwargs):

    # Two layer model for true velocity
    model = demo_model(preset, shape=shape, spacing=spacing,
                       space_order=space_order, nbl=nbl, **kwargs)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    return PureTtiWaveSolver(model, geometry, space_order=space_order,
                             **kwargs)


def run(shape=(50, 50, 50), spacing=(20.0, 20.0, 20.0), tn=250.0,
        autotune=False, space_order=4, nbl=10, preset='layers-tti',
        full_run=False, checkpointing=False, **kwargs):

    solver = tti_setup(shape=shape, spacing=spacing, tn=tn, space_order=space_order,
                       nbl=nbl, preset=preset, **kwargs)

    if full_run:
        info("--full was enabled, but so far, we only have implemented the" + "\n"
             + "Forward operator. Thus, it is the only operator used." + "\n")
        full_run = not full_run

    if checkpointing:
        info("Gradient modelling function is not yet ready. Checkpointing" + "\n"
             + "variable is disregarded." + "\n")
        checkpointing = not checkpointing

    info("Applying Forward")
    # Whether or not we save the whole time history. We only need the full wavefield
    # with 'save=True' if we compute the gradient without checkpointing, if we use
    # checkpointing, PyRevolve will take care of the time history
    save = full_run and not checkpointing
    # Define receiver geometry (spread across x, just below surface)
    rec, u, summary = solver.forward(save=save, autotune=autotune)

    plt.imshow(u.data[100,:,25,:])
    plt.show()
    return summary.gflopss, summary.oi, summary.timings, [rec, u]


@pytest.mark.parametrize('shape', [(51, 51), (16, 16, 16)])
def test_tti_stability(shape):
    spacing = tuple([20]*len(shape))
    _, _, _, [rec, _] = run(shape=shape, spacing=spacing,
                            tn=16000.0, nbl=0)
    assert np.isfinite(norm(rec))


if __name__ == "__main__":
    description = ("Example script to execute a TTI forward operator.")
    parser = seismic_args(description)
    args = parser.parse_args()

    # Switch to TTI kernel if input is acoustic kernel
    preset = 'layers-tti'

    # Preset parameters
    ndim = args.ndim
    shape = args.shape[:args.ndim]
    spacing = tuple(ndim * [20.0])
    tn = args.tn if args.tn > 0 else (750. if ndim < 3 else 1250.)

    run(shape=shape, spacing=spacing, nbl=args.nbl, tn=tn,
        space_order=args.space_order, autotune=args.autotune, dtype=args.dtype,
        opt=args.opt, preset=preset, checkpointing=args.checkpointing,
        full_run=args.full)
