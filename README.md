# OSCAR: configure and debug variational quantum algorithms (VQAs) efficiently
OSCAR leverages compressed sensing to reconstruct the landscape of variational quantum algorithms, using only a small fraction of all circuit executions needed for the entire landscape.

This is a package accompanying the paper [Enabling High Performance Debugging for Variational Quantum Algorithms using Compressed Sensing](https://doi.org/10.1145/3579371.3589044). For our original research implementation and record, please refer to [https://github.com/kunliu7/oscar/](https://github.com/kunliu7/oscar/). This repo is a rewrite as a user-friendly package, and some of the methods have been substantially improved in comparison to the version used in the paper.

## Install
```
pip install git+https://github.com/haoty/OSCAR
```

## Get Started
__The following walkthrough is also available as a [Jupyter notebook](https://github.com/HaoTy/OSCAR/blob/main/notebooks/get_started.ipynb).__

### Introduction

An "(energy) landscape" of a variational quantum algorithm (VQA) is the ensemble of objective function values over the parameter space, where each value is the expectation of measuring the problem Hamiltonian with the variational ansatz under the corresponding parameters. OSCAR exploits landscapes to provide VQA debugging featuers.

In OSCAR, the `oscar.Landscape` class uses a discretized grid over parameters in given ranges ([#Landscape](#landscape)), where the grid values are calculated by an `oscar.BaseExecutor` ([#Executor](#executor)). To speed up this grid generation process, OSCAR provides the option to approximate the grid values using only a small fraction of samples ([#Reconstruction](#reconstruction)). Additionally, OSCAR can interpolate the grid points to provide a continuous function approximating the landscape for instant optimization runs ([#Interpolation](#interpolation)). 

### Landscape

Define the landscape with parameter resolutions (granularity) and parameter bounds.
The order of the parameters corresponds to their order in the VQA ansatz definition. In the case of qiskit QAOA, the order is all betas and then all gammas.

```python
from oscar import Landscape
from math import pi

resolution = [64, 64]
bounds = [(-pi / 4, pi / 4), (-pi / 2, pi / 2)]
landscape = Landscape(resolution, bounds)
```

### Executor
#### Custom Executor

An executor can be easily constructed with a user-defined function that outputs the value of given input parameters.

```python
from collections.abc import Sequence
from oscar import CustomExecutor

def f(params: Sequence[float]) -> float:
    ...

custom_executor = CustomExecutor(f)
```

#### Qiskit Executor

OSCAR also provides an executor that works with Qiskit problem and VQA classes.
As an example, let's solve a 3-regular graph MaxCut problem with QAOA. First, define the problem of interest with `qiskit_optimization` (or `qiskit_finance`, `qiskit_nature`, or `docplex.mp` for more problems).

```python
import networkx as nx
from qiskit_optimization.applications import Maxcut

n = 8
graph = nx.random_regular_graph(3, n)
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
landscape.sample_and_run(qiskit_executor, sampling_fraction = 1 / 16)
```

Reconstruct the full landscape with a desired `oscar.BaseReconstructor` and visualize the reconstructed landscape.

```python
from oscar import BPDNReconstructor, plot_2d_landscape

landscape.reconstruct(BPDNReconstructor(solver = None)) # choose and config a desired cvxpy solver
figure = plot_2d_landscape(landscape, which_landscape="reconstructed")
```

![assets/get_started_1.png](assets/get_started_1.png?raw=true)

Run the true landscape and compare.

```python
# may take some time
landscape.run_all(qiskit_executor)
figure = plot_2d_landscape(landscape, which_landscape="true")
```

![assets/get_started_2.png](assets/get_started_2.png?raw=true)

### Interpolation
OSCAR can interpolate the grid points to get a continuous approximation of the landscape, which can in turn serve as an executor for optimizers and other purposes.

```python
from oscar import QiskitOptimizer, InterpolatedLandscapeExecutor
from qiskit.algorithms.optimizers import COBYLA

landscape.interpolate(method="slinear", fill_value=1)
itpl_executor = InterpolatedLandscapeExecutor(landscape)

def optimize_result(executor):
    trace, original_result = QiskitOptimizer(COBYLA(100, rhobeg=0.3)).run(
        itpl_executor, initial_point=[0.1, -0.1], bounds=bounds
    )
    trace.print_result()
    plot_2d_landscape(landscape, trace)

optimize_result(itpl_executor)
```

![assets/get_started_3.png](assets/get_started_3.png?raw=true)

Compare with the optimization where the values are calculated by actual circuit executions.

```python
optimize_result(qiskit_executor)
```

![assets/get_started_4.png](assets/get_started_4.png?raw=true)

We see that the results are very close, while the time for optimizing with the interpolated landscape is negligible compared to the actual optimization.

Landscapes can be easily saved for later retrieval.
```python
import numpy as np

filename = f"../data/landscapes/p=1-{n=}-{bounds}-{resolution}.pckl"
landscape.save(filename)
landscape = np.load(filename, allow_pickle=True)
```

## Roadmap
- Execution
    - Executors for other backends, e.g. Cirq.
- Landscape
    - Multi-source executions and noise compensation
    - Better visualization, especially for higher dimensions
    - Execution-hardness-aware parameter sampling
- Reconstruction
    - Avoid explicitly constructing the inverse DCT operator (by adding `scipy.optimize` or `cvxopt` implementations)
    - Other types of compressed sensing (e.g. total variation)
- Optimization
    - Directly interface scipy and nlopt optimizers
