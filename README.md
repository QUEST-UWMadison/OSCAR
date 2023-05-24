# OSCAR: cOmpressed Sensing based Cost lAndscape Reconstruction
OSCAR leverages compressed sensing to reconstruct the landscape of variational quantum algorithms, using only a small fraction of all circuit executions needed for the entire landscape.

This is a package accompanying the paper [TODO](). For our original research implementation and record, please refer to [this repo](https://github.com/kunliu7/oscar/).

## Install
```
pip install git+https://github.com/haoty/QSCAR
```

## Get Started
__The following walkthrough is also available as a [Jupyter notebook](https://github.com/HaoTy/OSCAR/blob/main/notebooks/get_started.ipynb).__
First, define the problem of interest with `qiskit_optimization`, `qiskit_finance`, or `docplex.mp`.


```python
import networkx as nx
from qiskit_optimization.applications import Maxcut

n = 8
graph = nx.random_regular_graph(3, n)
problem = Maxcut(graph).to_quadratic_program()
H, offset = problem.to_ising() # construct the Hamiltonian
```

Define the desired variational quantum algorithm with `qiskit`.
OSCAR supports both the old `VQE` and `QAOA` that are being deprecated and the new `VQE`, `SamplingVQE`, and `QAOA` that use the `Estimator` and `Sampler` primitives.


```python
# Old
from qiskit_aer import AerSimulator
from qiskit.algorithms import QAOA as OldQAOA
from qiskit.algorithms.optimizers import COBYLA

algorithm = OldQAOA(COBYLA(), quantum_instance=AerSimulator())
```


```python
# New
from qiskit_aer.primitives import Sampler
from qiskit.algorithms.minimum_eigensolvers import QAOA as NewQAOA

algorithm = NewQAOA(Sampler(), COBYLA())
```

Define the landscape with parameter resolutions (granularity) and parameter bounds.
The order of the parameters corresponds to their order in the VQA ansatz definition. In the case of qiskit QAOA, the order is all betas and then all gammas.


```python
from oscar import Landscape, QiskitExecutor, BPDNReconstructor
from math import pi

landscape = Landscape([64, 64], [(-pi / 4, pi / 4), (-pi / 2, pi / 2)])
```

Define an executor responsible for computing the landscape data with the previously constructed VQA and Hamiltonian and generate the sampled points.


```python
executor = QiskitExecutor(algorithm, H)
landscape.run_after_sample(executor, sampling_fraction = 1 / 16)
```


Reconstruct the full landscape with a desired Reconstructor and visualize the reconstructed landscape.


```python
import matplotlib.pyplot as plt

landscape.reconstruct(BPDNReconstructor(solver = None)) # choose and config a desired cvxpy solver
plt.imshow(landscape.reconstructed_landscape)
```


    
![https://github.com/HaoTy/OSCAR/figs/get_started_1.png](https://github.com/HaoTy/OSCAR/blob/main/figs/get_started_1.png?raw=true)



Run the true landscape and compare.


```python
landscape.run_all(executor)
plt.imshow(landscape.true_landscape)
```

    
![https://github.com/HaoTy/OSCAR/figs/get_started_2.png](https://github.com/HaoTy/OSCAR/blob/main/figs/get_started_2.png?raw=true)




## Possible Enhancements
- Execution
    - Executors for other backends, e.g. Cirq.
- Landscape
    - Handle parallel/distributed executions and avoid rerun the same parameters
    - Interpolation and optimization on reconstructed landscape
    - Better visualization
    - Enhancements for constrained problems
        - Linear combinations of landscapes for deciding a proper penalty factor for the Lagrange method
        - Reconstructions for approximation ratio and in-constraint probability
- Reconstruction
    - Avoid explicitly constructing the inverse DCT operator (by adding `scipy.optimize` or `cvxopt` implementations)
    - Other types of compressed sensing (e.g. total variation)
- Misc.
    - Command line interface
