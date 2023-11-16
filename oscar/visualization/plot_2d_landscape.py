from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..landscape import Landscape
from ..optimization import Trace


def plot_2d_landscape(
    landscape: Landscape,
    trace: Trace | None = None,
    trace_plot_params: tuple[int, int] = (0, 1),
    show: bool = True,
    figure: Figure | None = None,
) -> Figure:
    if landscape.num_params != 2:
        raise ValueError("Landscape must be two-dimensional.")
    fig = plt.figure(figure)
    plt.imshow(
        landscape.landscape.T,
        extent=landscape.param_bounds.flat,
        origin="lower",
        interpolation="none",
    )
    plt.colorbar()
    if trace is not None:
        trace_plot_params = np.array(trace_plot_params)
        plt.plot(
            *np.array(trace.params_trace).T[trace_plot_params],
            alpha=0.75,
            label="optimizer trace",
            color="red",
            linewidth=1,
        )
        plt.scatter(
            *np.array(trace.params_trace).T[trace_plot_params],
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
