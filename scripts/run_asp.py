import numpy as np
from qore import Mine, ASP, QAOA, VQE
from qore.utils import measure_operator, get_bitstring_probabilities, identity
from qiskit.utils import algorithm_globals, QuantumInstance, quantum_instance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.opflow import I, Z, Plus
from qore.benchmark import Benchmark


# define a callback function
def analysis(circ, iter):
    print(f"--- Iter {iter} ---")
    x = get_bitstring_probabilities(circ)
    bitstr, prob = max(x.items(), key=lambda item: item[1])
    print(f"The most probable configuration and the corresponding probability: {bitstr, prob}")
    qmine.plot_mine_state(bitstr)


if __name__ == "__main__":

    penalty = 10.0
    qmine = Mine(np.array([[-2.0, 3.0, 1.0], [float("inf"), 5.0, float("inf")]]))

    qmine = Mine(
        np.array(
            [
                [-2.0, 3.0, -1.0, -2.0, -1.0],
                [float("inf"), 1.0, -5.0, 10.0, float("inf")],
                [float("inf"), float("inf"), 4.0, float("inf"), float("inf")],
            ]
        )
    )

    qmine.plot_mine()

    # algorithm_globals.random_seed = 1953
    algorithm_globals.massive = True

    evol_time = 10
    nsteps = 20

    asp = ASP(
        evol_time,
        nsteps,
        # callback=analysis,
        # callback_freq=5,
        quantum_instance=QasmSimulator(),
    )
    
    res = qmine.solve(asp, False, False)
    print(res)
