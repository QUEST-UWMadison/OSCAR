from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import product
from typing import TYPE_CHECKING, Any
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer

if TYPE_CHECKING:
    from ..execution.base_executor import BaseExecutor
    from .trace import Trace


class HyperparameterSet(dict):
    def __init__(self, optimizer: BaseOptimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer: BaseOptimizer = optimizer

    @property
    def shape(self) -> tuple[int]:
        return tuple(
            len(self.values()),
        )


class HyperparameterGrid(dict):
    def __init__(self, optimizer: BaseOptimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer: BaseOptimizer = optimizer

    @property
    def method(self) -> str:
        return self.optimizer.name()

    @property
    def shape(self) -> tuple[int]:
        return tuple(len(values) for values in self.values())

    def generate_hyperparameter_sets(self) -> Iterator[HyperparameterSet]:
        for key, value in self.items():
            # turn dict of lists into list of dicts
            if isinstance(value, dict):
                self[key] = (dict(zip(value, comb)) for comb in product(*value.values()))
        for values in product(*self.values()):
            yield HyperparameterSet(deepcopy(self.optimizer), dict(zip(self.keys(), values)))


class HyperparameterTuner:
    def __init__(self, configs: Sequence[HyperparameterGrid | Iterable[HyperparameterSet]]):
        self.configs: Sequence[HyperparameterGrid | Iterable[HyperparameterSet]] = configs
        self.shapes: dict[str, tuple[int]] = {}
        self.traces: dict[str, Trace] = {}
        self.results: dict[str, Any] = {}
        for config_set in self.configs:
            self.shapes[config_set.method] = config_set.shape

    @property
    def methods(self) -> tuple[str]:
        return tuple(config_set.method for config_set in self.configs)

    def run(
        self,
        executors: BaseExecutor | Sequence[BaseExecutor],
    ) -> tuple[dict[str, list[Trace]], dict[str, list[Any]]]:
        if not isinstance(executors, Sequence):
            executors = [executors] * len(self.configs)

        traces, results = {}, {}
        for config_set, executor in zip(self.configs, executors):
            method = config_set.method
            self.shapes[method] = config_set.shape
            traces[method] = []
            results[method] = []
            if isinstance(config_set, HyperparameterGrid):
                config_set = config_set.generate_hyperparameter_sets()
            for config in config_set:
                trace, result = config.optimizer.run(executor, **config)
                traces[method].append(trace)
                results[method].append(result)
        self.traces = traces
        self.results = results
        return traces, results

    def process_results(
        self, function: Callable[[str, int, Trace, Any], Any], reshape: bool = True
    ) -> dict[str, NDArray[Any]]:
        ret = {}
        for method in self.methods:
            ret[method] = []
            for i, (trace, result) in enumerate(zip(self.traces[method], self.results[method])):
                ret[method].append(function(method, i, trace, result))
            ret[method] = np.array(ret[method])
            if reshape:
                ret[method] = ret[method].reshape(*self.shapes[method], -1)
        return ret
