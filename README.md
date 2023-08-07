# OSCAR: configure and debug variational quantum algorithms (VQAs) efficiently
OSCAR leverages compressed sensing to reconstruct the landscape of variational quantum algorithms, using only a small fraction of all circuit executions needed for the entire landscape. In addition, by interpolating the discrete landscape, OSCAR can benchmark thousands of optimization configurations in an instant.

This is a package accompanying the paper [Enabling High Performance Debugging for Variational Quantum Algorithms using Compressed Sensing](https://arxiv.org/abs/2308.03213). For our original research implementation and record, please refer to [https://github.com/kunliu7/oscar/](https://github.com/kunliu7/oscar/). This repo is a rewrite as a user-friendly package, and some of the methods have been substantially improved in comparison to the version used in the paper.

## Install
```
pip install git+https://github.com/haoty/OSCAR
```

## Get Started
__The following walkthrough is also available as a [Jupyter notebook](https://github.com/HaoTy/OSCAR/blob/main/notebooks/get_started.ipynb).__

### Introduction

An "(energy) landscape" of a variational quantum algorithm (VQA) is the ensemble of objective function values over the parameter space, where each value is the expectation of measuring the problem Hamiltonian with the variational ansatz under the corresponding parameters. OSCAR exploits landscapes to provide VQA debugging featuers.

In OSCAR, the `oscar.Landscape` class uses a discretized grid over parameters in given ranges ([#Landscape](#landscape)), where the grid values are calculated by an `oscar.BaseExecutor` ([#Executor](#executor)). To speed up this grid generation process, OSCAR provides the option to approximate the grid values using only a small fraction of samples ([#Reconstruction](#reconstruction)). Additionally, OSCAR can interpolate the grid points to provide a continuous function approximating the landscape for instant optimization runs ([#Interpolation](#interpolation)), thus enabling highly efficient [#Optimization configuration benchmarking](#optimization-configuration-benchmarking) for choosing optimizers, their hyperparameters, initialization strategies, and more.

### Landscape

Define the landscape with parameter resolutions (granularity) and parameter bounds.
The order of the parameters corresponds to their order in the VQA ansatz definition. In the case of qiskit QAOA, the order is all betas and then all gammas.


```python
from oscar import Landscape
from math import pi

resolution = [64, 64]
bounds = [(-pi / 4, pi / 4), (0, pi / 2)]
landscape = Landscape(resolution, bounds)
```

### Executor
#### Custom Executor

An executor can be easily constructed with a user-defined function that outputs the value of given input parameters.


```python
from __future__ import annotations
from collections.abc import Sequence
from oscar import CustomExecutor

def f(params: Sequence[float]) -> float:
    ...

custom_executor = CustomExecutor(f)
```

#### Qiskit Executor

OSCAR also provides an executor that works with Qiskit problem and VQA classes.
As an example, let's solve a 3-regular graph MaxCut problem with QAOA. First, define the problem of interest with `qiskit_optimization` (or `qiskit_finance`, `qiskit_nature`, or `docplex.mp` for more problems). These packages need to be manually installed.


```python
import networkx as nx
from qiskit_optimization.applications import Maxcut

n = 10
graph = nx.random_regular_graph(3, n, 42)
problem = Maxcut(graph).to_quadratic_program()
H, offset = problem.to_ising() # construct the Hamiltonian
```

Define the desired Qiskit VQA.
OSCAR supports both the old `VQE` and `QAOA` that are being deprecated and the new `VQE`, `SamplingVQE`, and `QAOA` that use the `Estimator` and `Sampler` primitives.


```python
# New
from qiskit_aer.primitives import Sampler
from qiskit.algorithms.minimum_eigensolvers import QAOA as NewQAOA
from qiskit.algorithms.optimizers import COBYLA

algorithm = NewQAOA(Sampler(), COBYLA())
```


```python
# Old
from qiskit_aer import AerSimulator
from qiskit.algorithms import QAOA as OldQAOA
from qiskit.algorithms.optimizers import COBYLA

algorithm = OldQAOA(COBYLA(), quantum_instance=AerSimulator())
```

Define the executor responsible for computing the landscape data with the previously constructed VQA and Hamiltonian and generate the sampled points.


```python
from oscar import QiskitExecutor

qiskit_executor = QiskitExecutor(algorithm, H)
```

### Reconstruction

Sample a few points on the grid and get their value using our previously-defined executor.


```python
_ = landscape.sample_and_run(qiskit_executor, sampling_fraction = 1 / 16, rng = 42)
```

Reconstruct the full landscape with a desired `oscar.BaseReconstructor` and visualize the reconstructed landscape.


```python
from oscar import BPDNReconstructor, plot_2d_landscape, BPReconstructor

landscape.reconstruct(BPReconstructor(solver = None)) # choose and config a desired cvxpy solver
figure = plot_2d_landscape(landscape)
```



![assets/get_started_1.png](assets/get_started_1.png?raw=true)



Uncomment to run the true landscape and compare. (May take some time)


```python
# may take some time
# exact_landscape = Landscape.like(landscape)
# exact_landscape.run_all(qiskit_executor)
# figure = plot_2d_landscape(exact_landscape)
```



![assets/get_started_1.png](assets/get_started_2.png?raw=true)


Landscapes can be easily saved for later retrieval.


```python
filename = f"../data/landscapes/p=1-{n=}-{bounds}-{resolution}.pckl"
landscape.save(filename)
landscape = Landscape.load(filename)
```


### Interpolation
OSCAR can interpolate the grid points to get a continuous approximation of the landscape, which can in turn serve as an executor for optimizers and other purposes.


```python
from oscar import InterpolatedLandscapeExecutor, NLoptOptimizer
from qiskit.algorithms.optimizers import COBYLA

landscape.interpolate(method="slinear", fill_value=3)
itpl_executor = InterpolatedLandscapeExecutor(landscape)

def optimize_and_show(executor):
    trace, original_result = NLoptOptimizer("LN_BOBYQA").run(
        executor, initial_point=[0.3, 0.5], bounds=bounds, xtol_abs=1e-8, initial_step=0.3
    )
    trace.print_result()
    plot_2d_landscape(landscape, trace)

optimize_and_show(itpl_executor)
```

    Total time: 0.0212554931640625
    Optimal parameters reported:  [-0.38839545  0.50520292]
    Optimal value reported:  -2.643549075727872
    Number of evaluations:  43




![assets/get_started_1.png](assets/get_started_3.png?raw=true)



Compare with the optimization where the values are calculated by actual circuit executions.


```python
optimize_and_show(qiskit_executor)
```

    Total time: 2.5728940963745117
    Optimal parameters reported:  [-0.34238969  0.51203497]
    Optimal value reported:  -2.59765625
    Number of evaluations:  43




![assets/get_started_1.png](assets/get_started_4.png?raw=true)



We see that the results are very close, while the time for optimizing with the interpolated landscape is negligible compared to the actual optimization, especially when the problem size is large.

### Optimization Configuration Benchmarking
We can specify combinations of hyperparameter values with `oscar.HyperparameterGrid` or `oscar.HyperparameterSet` and then utilize `oscar.HyperparameterTuner` to conveniently do a grid search over all combinations. If a landscape object is available, we can take advantage of the interpolated executor to reduce the grid search time to seconds.


```python
from math import prod
from time import time
from oscar import HyperparameterTuner, HyperparameterGrid, result_metrics

x0_pool = [(0, 0.6), (0.4, 0.6), (0, 1.2), (-0.4, 1.2), (0.4, 1.2)]
maxfev_pool = [10, 30, 50]
initial_step_pool = [0.001, 0.01, 0.1]
configs = [
    HyperparameterGrid(
        NLoptOptimizer("LN_COBYLA"),
        initial_point=x0_pool,
        maxiter=maxfev_pool,
        initial_step=initial_step_pool,
        bounds=[bounds],
        ftol_rel=[1e-14],
    ),
    HyperparameterGrid(
        NLoptOptimizer("LN_BOBYQA"),
        initial_point=x0_pool,
        maxeval=maxfev_pool,
        initial_step=initial_step_pool,
        bounds=[bounds],
        ftol_rel=[1e-14],
    )
]

tuner = HyperparameterTuner(configs)
print(f"Running {sum(prod(config.shape) for config in configs)} optimizations...")
start = time()
tuner.run(itpl_executor)
print(f"...in {time() - start:.2f} seconds.")
```

    Running 90 optimizations...
    ...in 2.67 seconds.


Print out top optimizer configurations using the optimizer reported optimal value (energy) as the metric.


```python
import numpy as np

result = tuner.process_results(result_metrics.optimal_value())
for config in configs:
    method = config.method
    top_config_idices = np.argsort(result[method].flat)[:5]
    print(f"Top configs for {method}:")
    for energy, config_str in zip(result[method].flat[top_config_idices], config.interpret(top_config_idices)):
        print(f"    Energy: {energy}  Config: {config_str}")
```

    Top configs for LN_COBYLA (NLopt):
        Energy: -2.685759939394269  Config: ['initial_point=(-0.4, 1.2)', 'maxiter=50', 'initial_step=0.1']
        Energy: -2.685759939394269  Config: ['initial_point=(-0.4, 1.2)', 'maxiter=30', 'initial_step=0.1']
        Energy: -2.685759939394269  Config: ['initial_point=(-0.4, 1.2)', 'maxiter=10', 'initial_step=0.1']
        Energy: -2.663261029214742  Config: ['initial_point=(0, 0.6)', 'maxiter=30', 'initial_step=0.001']
        Energy: -2.663261029214742  Config: ['initial_point=(0, 0.6)', 'maxiter=50', 'initial_step=0.001']
    Top configs for LN_BOBYQA (NLopt):
        Energy: -2.6857599393943636  Config: ['initial_point=(0, 0.6)', 'maxeval=50', 'initial_step=0.1']
        Energy: -2.685759939394363  Config: ['initial_point=(-0.4, 1.2)', 'maxeval=50', 'initial_step=0.1']
        Energy: -2.6857599389829905  Config: ['initial_point=(0, 0.6)', 'maxeval=30', 'initial_step=0.1']
        Energy: -2.6857563910263353  Config: ['initial_point=(-0.4, 1.2)', 'maxeval=30', 'initial_step=0.1']
        Energy: -2.6632610292147447  Config: ['initial_point=(0, 1.2)', 'maxeval=50', 'initial_step=0.1']


## Roadmap
- Docs and tests
- Execution
    - Executors for other backends, e.g. Cirq.
- Landscape
    - Better visualization, especially for higher dimensions
    - Execution-hardness-aware parameter sampling
    - Support for irregular grid (integrate with `HyperparameterTuner`)
- Reconstruction
    - Avoid explicitly constructing the inverse DCT operator (by adding `scipy.optimize` or `cvxopt` implementations)
    - Other types of compressed sensing (e.g. total variation)
- Optimization
    - Directly interface scipy optimizers
    - Support the association of hyperparameters in tuner
