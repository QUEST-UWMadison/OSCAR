from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from ..landscape import Landscape
from ..optimization import Trace


def plot_2d_landscape(
    landscape: Landscape,
    trace: Trace | None = None,
    plot_optimum: Literal["min", "max", False, None] | tuple[float, float] = "min",
    show: bool = True,
    figure: Figure | Axes | None = None,
    custom_plot_landscape_func: Callable | None = None,
    custom_plot_trace_func: Callable | None = None,
    custom_plot_optimum_func: Callable | None = None,
) -> Figure | Axes:
    if landscape.num_params != 2:
        raise ValueError("Landscape must be two-dimensional.")
    landscape = landscape.to_dense()
    fig = plt.figure(figure)
    if custom_plot_landscape_func is None:
        plt.imshow(
            landscape.landscape.to_numpy().T,
            extent=landscape.param_bounds.flat,
            origin="lower",
            interpolation="none",
        )
        plt.colorbar()
    else:
        custom_plot_landscape_func(landscape.landscape.to_numpy().T)
    if trace is not None:
        params_trace = np.asarray(trace.params_trace).T
        if custom_plot_trace_func is None:
            plt.plot(
                *params_trace,
                alpha=0.75,
                label="optimizer trace",
                color="red",
                linewidth=1,
            )
            plt.scatter(
                *params_trace,
                marker="x",
                alpha=0.75,
                label="optimizer query",
                c=range(len(trace.params_trace)),
                cmap="autumn",
                s=15,
            )
        else:
            custom_plot_trace_func(*params_trace)
    
    if plot_optimum is None and trace is not None:
        plot_optimum = trace.optimization_type
    if plot_optimum is not False:
        if plot_optimum == "min":
            optimum = landscape.argmin()
        elif plot_optimum == "max":
            optimum = landscape.argmax()
        else:
            optimum = plot_optimum
        if custom_plot_optimum_func is None:
            plt.scatter(
                *optimum,
                marker="*",
                color="white",
                s=20,
                label="grid optimum",
            )
        else:
            custom_plot_optimum_func(optimum)
    plt.legend()
    if show:
        plt.show()
    return fig
