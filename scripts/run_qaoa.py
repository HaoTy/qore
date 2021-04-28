import numpy as np
from qore import Mine, ASP, QAOA, VQE
from qore.utils import measure_operator, get_bitstring_probabilities, identity
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA, SNOBFIT
from qiskit.providers.aer import QasmSimulator


if __name__ == "__main__":

    penalty = 10.0
    qmine = Mine(np.array([[-2.0, 3.0, 1.0], [float("inf"), 5.0, float("inf")]]))

    penalty = 12.0
    qmine = Mine(
        np.array(
            [
                [-2.0, 3.0, -1.0, -2.0, -1.0],
                [float("inf"), 1.0, -5.0, 10.0, float("inf")],
                [float("inf"), float("inf"), 4.0, float("inf"), float("inf")],
            ]
        )
    )
    # penalty = 10.0
    # qmine = Mine('./mine_configs/30sites.txt')

    qmine.plot_mine()

    # algorithm_globals.random_seed = 1953

    p = 3

    qaoa = QAOA(
        optimizer=COBYLA(),
        quantum_instance=QasmSimulator(),
        reps=p
    )

    res = qmine.solve(qaoa, None, False)
    print(res)
