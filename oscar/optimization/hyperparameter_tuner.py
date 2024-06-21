from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import product
from typing import TYPE_CHECKING, Any, Type

import numpy as np
from numpy.typing import NDArray

from .base_optimizer import BaseOptimizer

if TYPE_CHECKING:
    from .trace import Trace


class HyperparameterSet(dict):
    def __init__(self, optimizer_cls: Type[BaseOptimizer], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_cls: Type[BaseOptimizer] = optimizer_cls

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            len(self.values()),
        )


class HyperparameterGrid(dict):
    def __init__(self, optimizer_cls: Type[BaseOptimizer] | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_cls: Type[BaseOptimizer] | None = optimizer_cls

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.values())

    def generate_hyperparameter_sets(self) -> Iterator[HyperparameterSet]:
        for key, value in self.items():
            # turn dict of lists into list of dicts
            if isinstance(value, dict):
                self[key] = [dict(zip(value, comb)) for comb in product(*value.values())]
        for values in product(*self.values()):
            yield HyperparameterSet(self.optimizer_cls, dict(zip(self.keys(), values)))

    def interpret(
        self, indices: int | Sequence[int], ignore_fixed_config: bool = True
    ) -> list[str] | list[list[str]]:
        single_idx = False
        if isinstance(indices, int):
            single_idx = True
            indices = [indices]
        ret = []
        for index in np.asarray(np.unravel_index(indices, self.shape)).T:
            ret.append([])
            for idx, (key, value) in zip(index, self.items()):
                if not ignore_fixed_config or len(value) != 1:
                    ret[-1].append(f"{key}={value[idx]}")
        return ret[0] if single_idx else ret


class HyperparameterTuner:
    def __init__(self, configs: Sequence[HyperparameterGrid | Iterable[HyperparameterSet]]):
        self.configs: Sequence[HyperparameterGrid | Iterable[HyperparameterSet]] = configs
        self.shapes: list[tuple[int, ...]] = {}
        self.traces: list[Trace] = []
        self.results: list[Any] = []

    def run(
        self,
        run_configs: HyperparameterGrid | Iterable[HyperparameterSet],
        bind_configs: bool = False,
    ) -> tuple[dict[str, list[Trace]], dict[str, list[Any]]]:
        if bind_configs:
            loop_func = zip
            self.shapes = [config_set.shape for config_set in self.configs]
        else:
            loop_func = product
            self.shapes = [config_set.shape + run_configs.shape for config_set in self.configs]

        traces, results = [], []
        for config_set in self.configs:
            traces.append([])
            results.append([])
            if isinstance(config_set, HyperparameterGrid):
                config_set = config_set.generate_hyperparameter_sets()
            if isinstance(run_configs, HyperparameterGrid):
                run_config_set = run_configs.generate_hyperparameter_sets()
            for config, run_config in loop_func(config_set, run_config_set):
                trace, result = config.optimizer_cls(**config).run(**run_config)
                traces[-1].append(trace)
                results[-1].append(result)
        self.traces = traces
        self.results = results
        return traces, results

    def process_results(
        self, function: Callable[[int, int, Trace, Any], Any], reshape: bool = True
    ) -> list[Any]:
        ret = []
        for i, (traces, results) in enumerate(zip(self.traces, self.results)):
            ret.append([])
            for j, (trace, result) in enumerate(zip(traces, results)):
                ret[-1].append(function(i, j, trace, result))
            if reshape:
                ret[-1] = np.array(ret[-1]).reshape(*self.shapes[i], -1).squeeze()
        return ret

    def interpret(
        self,
        indices: int | Sequence[int],
        which_config: int = 0,
        run_configs: HyperparameterGrid | Iterable[HyperparameterSet] = None,
        ignore_fixed_config: bool = True,
    ) -> list[str] | list[list[str]]:
        if isinstance(indices, int):
            indices = [indices]
        configs = [self.configs[which_config]]
        if run_configs is not None:
            configs.append(run_configs)
        split = len(np.squeeze(configs[0].shape))
        interpretations = []
        for config in configs:
            if config is run_configs:
                slice_ = slice(split, None)
            else:
                slice_ = slice(None, split)
            interpretations.append(config.interpret(
                np.ravel_multi_index(
                    np.unravel_index(indices, self.shapes[which_config])[slice_],
                    config.shape,
                ),
                ignore_fixed_config,
            ))
        return [config + run_config for config, run_config in zip(*interpretations)]
