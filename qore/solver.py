from typing import Optional, Union, Dict, List, Callable
import numpy as np

from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA

from qore import Mine, ASP, QAOA, Pseudoflow
from qore.utils import measure_operator, get_bitstring_probabilities


class Solver():
    """
    Abstract solver for an open pit mining circuit.
    """

    def __init__(self, model: Mine) -> None:
        self._model = model

    def solve(self) -> Dict:
        """
        Return a result dict.
        """
        pass


class PseudoflowSolver(Solver):
    """
    Solver for the Pseudoflow algorithm.
    """

    def __init__(self, model: Mine) -> None:
        super().__init__(model)

    def solve(self) -> Dict:
        graph, source, sink = self._model.gen_pseudoflow_graph()
        pf = Pseudoflow(graph, source, sink)
        return pf.run()


class QuantumSolver(Solver):
    """
    Abstract quantum solver
    """

    def __init__(self,
                 model: Mine,
                 random_seed,
                 backend,
                 shots
                 ) -> None:
        super().__init__(model)
        self.circuit = None
        algorithm_globals.random_seed = random_seed
        self.backend = backend
        self.instance = QuantumInstance(backend=self.backend)
        self.shots = shots

    def solve(self) -> Dict:
        """
        Return a result dict.
        """
        pass


class ASPSolver(QuantumSolver):
    """
    Solver for the ASP algorithm.
    """

    def __init__(self,
                 model: Mine,
                 random_seed: Optional[int] = 1953,
                 backend=Aer.get_backend('statevector_simulator'),
                 shots: Optional[int] = 1024
                 ) -> None:
        super().__init__(model, random_seed, backend, shots)

    def solve(self, penalty, **kwargs) -> Dict:
        self.circuit = ASP(self._model.gen_Hamiltonian(penalty=penalty),
                           **kwargs).construct_circuit()

        x = get_bitstring_probabilities(
            self.circuit, self.instance, shots=self.shots)
        bitstr, prob = max(x.items(), key=lambda item: item[1])

        res = {"opt_config": bitstr,
               "opt_config_prob": prob,
               "profit_avg": measure_operator(self._model.Hp, self.circuit, self.instance),
               "violation_avg": measure_operator(self._model.Hs, self.circuit, self.instance),
               "ground_state": x,
               }

        return res


class QAOASolver(QuantumSolver):
    """
    Solver for the ASP algorithm.
    """

    def __init__(self,
                 model: Mine,
                 optimizer=COBYLA(),
                 p=4,
                 random_seed: Optional[int] = 1953,
                 backend=Aer.get_backend('statevector_simulator'),
                 shots: Optional[int] = 1024
                 ) -> None:
        super().__init__(model, random_seed, backend, shots)
        self.optimizer = optimizer
        self.p = p

    def solve(self, penalty, **kwargs) -> Dict:
        qaoa = QAOA(
            optimizer=self.optimizer,
            quantum_instance=self.instance,
            reps=self.p)
        qaoa.compute_minimum_eigenvalue(
            self._model.gen_Hamiltonian(penalty=penalty))
        self.circuit = qaoa.get_optimal_circuit()

        x = get_bitstring_probabilities(
            self.circuit, self.instance, shots=self.shots)
        bitstr, prob = max(x.items(), key=lambda item: item[1])

        res = {"opt_config": bitstr,
               "opt_config_prob": prob,
               "profit_avg": measure_operator(self._model.Hp, self.circuit, self.instance),
               "violation_avg": measure_operator(self._model.Hs, self.circuit, self.instance),
               "ground_state": x,
               }

        return res


class FragmentationSolver(QuantumSolver):
    """
        Generalized fragmented ASP solver.
        """

    def __init__(self,
                 model: Mine,
                 random_seed: Optional[int] = 1953,
                 backend=Aer.get_backend('statevector_simulator'),
                 shots: Optional[int] = 1024,
                 ) -> None:
        super().__init__(model, random_seed, backend, shots)

    def calc_frag_H(self):
        pass

    def solve(self) -> Dict:
        pass
