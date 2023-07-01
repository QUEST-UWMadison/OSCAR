from __future__ import annotations

from collections.abc import Sequence

from qiskit.algorithms import VQE as OldVQE
from qiskit.algorithms.minimum_eigensolvers import VQE, SamplingVQE
from qiskit.opflow import OperatorBase
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_executor import BaseExecutor


class QiskitExecutor(BaseExecutor):
    def __init__(
        self, algorithm: OldVQE | VQE | SamplingVQE, operator: OperatorBase | BaseOperator
    ) -> None:
        self.algorithm: OldVQE | VQE | SamplingVQE = algorithm
        self.operator: OperatorBase | BaseOperator = operator
        algorithm._check_operator_ansatz(operator)
        if isinstance(algorithm, OldVQE):
            self.evaluate_energy = algorithm.get_energy_evaluation(operator)
        else:
            if isinstance(algorithm, SamplingVQE):
                if len(algorithm.ansatz.clbits) > 0:
                    algorithm.ansatz.remove_final_measurements()
                algorithm.ansatz.measure_all()
            self.evaluate_energy = algorithm._get_evaluate_energy(
                ansatz=algorithm.ansatz, operator=operator
            )

    def _run(self, params: Sequence[float]) -> float:
        return self.evaluate_energy(params)
