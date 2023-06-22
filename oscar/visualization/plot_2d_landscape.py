from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..landscape import Landscape
from ..optimization import Trace


def plot_2d_landscape(
    landscape: Landscape,
    trace: Trace | None = None,
    which_landscape: Literal["true", "reconstructed", "auto"] = "auto",
    show: bool = True,
    figure: Figure | None = None,
) -> Figure:
    if landscape.num_params != 2:
        raise ValueError("Landscape must be two-dimensional.")
    landscape_array = landscape._get_landscape(which_landscape)
    fig = plt.figure(figure)
    plt.imshow(
        landscape_array.T,
        extent=landscape.param_bounds.flat,
        origin="lower",
        interpolation="none",
    )
    # Can't tell the order of gamma and beta, or really, if it's QAOA
    # plt.xlabel("β")
    # plt.ylabel("γ", rotation=0, va="center")
    plt.colorbar()
    if trace is not None:
        plt.plot(
            *np.array(trace.params_trace).T,
            alpha=0.75,
            label="optimizer trace",
            color="red",
            linewidth=1,
        )
        plt.scatter(
            *np.array(trace.params_trace).T,
            marker="x",
            alpha=0.75,
            label="optimizer query",
            c=range(len(trace.params_trace)),
            cmap="autumn",
            s=15,
        )
    lower_bounds, upper_bounds = landscape.param_bounds.T
    plt.scatter(
        *(
            landscape._unravel_index([np.argmin(landscape_array)])
            / landscape.param_resolutions[:, np.newaxis]
            * (upper_bounds - lower_bounds)[:, np.newaxis]
            + lower_bounds[:, np.newaxis]
        ),
        marker="*",
        color="white",
        s=20,
        label="grid minima",
    )
    plt.legend()
    if show:
        plt.show()
    return fig
