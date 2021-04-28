import numpy as np
from qore import Mine, ASP, QAOA, VQE, ExactDiagonalization
from qore.utils import measure_operator, get_bitstring_probabilities, identity
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA, SPSA, AQGD
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import MatrixOp, I


if __name__ == "__main__":

    penalty = 10.0
    qmine = Mine(np.array([[-2.0, 3.0, 1.0], [float("inf"), 5.0, float("inf")]]))

    penalty = 11.0
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

    # algorithm_globals.random_seed = 1

    vqe = VQE(
        ansatz=EfficientSU2(
            qmine.nqubits,
            su2_gates=["ry"],
            entanglement="full",
            reps=3,
            insert_barriers=True,
        ),
        optimizer=COBYLA(),
        quantum_instance=QasmSimulator(),
    )

    res = qmine.solve(vqe, False, False)
    print(res)
