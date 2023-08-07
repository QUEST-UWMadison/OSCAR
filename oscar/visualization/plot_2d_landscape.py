from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..landscape import Landscape
from ..optimization import Trace


def plot_2d_landscape(
    landscape: Landscape,
    trace: Trace | None = None,
    show: bool = True,
    figure: Figure | None = None,
) -> Figure:
    if landscape.num_params != 2:
        raise ValueError("Landscape must be two-dimensional.")
    landscape_array = landscape.landscape
    fig = plt.figure(figure)
    plt.imshow(
        landscape_array.T,
        extent=landscape.param_bounds.flat,
        origin="lower",
        interpolation="none",
    )
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
    plt.scatter(
        *landscape.optimal_params,
        marker="*",
        color="white",
        s=20,
        label="grid minima",
    )
    plt.legend()
    if show:
        plt.show()
    return fig
