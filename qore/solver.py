from typing import Optional, Union, Dict, List, Callable
from abc import ABC, abstractmethod

import numpy as np
from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms.optimizers import COBYLA, Optimizer
from qiskit.providers import BaseBackend, Backend
import networkx as nx

from qore import Mine, ASP, QAOA, Pseudoflow
from qore.utils import measure_operator, get_bitstring_probabilities


class Solver(ABC):
    """
    Abstract solver for an open pit mining circuit.
    """

    def __init__(self, model: Mine) -> None:
        self._model = model
        self.result = None

    @abstractmethod
    def solve(self) -> Dict:
        """
        Return a result dict.
        """
        raise NotImplementedError()


class PseudoflowSolver(Solver):
    """
    Solver for the Pseudoflow algorithm.
    """

    def __init__(self, model: Mine) -> None:
        super().__init__(model)

    def solve(self) -> Dict:
        self.result = Pseudoflow(*self.gen_pseudoflow_graph()).run()
        return self.result

    def gen_pseudoflow_graph(
        self, MAX_FLOW: int = 1000000
    ) -> tuple[nx.DiGraph, int, int]:
        G = nx.DiGraph()
        G.add_nodes_from(np.arange(self._model.nqubits))
        source = -1
        sink = self._model.nqubits

        for p in self._model.graph:
            for c in self._model.graph[p]:
                G.add_edge(p, c, const=MAX_FLOW)

            if self._model.dat[self._model.idx2cord[p]] >= 0:
                G.add_edge(source, p, const=self._model.dat[self._model.idx2cord[p]])
            else:
                G.add_edge(p, sink, const=-self._model.dat[self._model.idx2cord[p]])

        return G, source, sink


class QuantumSolver(Solver):
    """
    Abstract quantum solver
    """

    def __init__(
        self,
        model: Mine,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        super().__init__(model)
        self.circuit = None
        self.quantum_instance = quantum_instance or QuantumInstance(
            Aer.get_backend("statevector_simulator")
        )

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """ Returns quantum instance. """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Union[QuantumInstance, BaseBackend, Backend]
    ) -> None:
        """ Sets quantum instance. """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance


class ASPSolver(QuantumSolver):
    """
    Solver for the ASP algorithm.
    """

    def __init__(
        self,
        model: Mine,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        super().__init__(model, quantum_instance)

    def solve(self, penalty: float, **kwargs) -> Dict:
        self.circuit = ASP(
            self._model.gen_Hamiltonian(penalty=penalty), **kwargs
        ).construct_circuit()

        x = get_bitstring_probabilities(
            self.circuit, self.quantum_instance
        )
        bitstr, prob = max(x.items(), key=lambda item: item[1])

        self.result = {
            "opt_config": bitstr,
            "opt_config_prob": prob,
            "profit_avg": measure_operator(
                self._model.Hp, self.circuit, self.quantum_instance
            ),
            "violation_avg": measure_operator(
                self._model.Hs, self.circuit, self.quantum_instance
            ),
            "ground_state": x,
        }

        return self.result


class QAOASolver(QuantumSolver):
    """
    Solver for the ASP algorithm.
    """

    def __init__(
        self,
        model: Mine,
        optimizer: Optional[Optimizer] = None,
        p: Optional[int] = 4,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        super().__init__(model, quantum_instance)
        self.optimizer = optimizer or COBYLA()
        self.p = p

    def solve(self, penalty: float, **kwargs) -> Dict:
        qaoa = QAOA(
            optimizer=self.optimizer,
            quantum_instance=self.quantum_instance,
            reps=self.p,
            **kwargs
        )
        qaoa.compute_minimum_eigenvalue(self._model.gen_Hamiltonian(penalty=penalty))
        self.circuit = qaoa.get_optimal_circuit()

        x = get_bitstring_probabilities(
            self.circuit, self.quantum_instance
        )
        bitstr, prob = max(x.items(), key=lambda item: item[1])

        self.result = {
            "opt_config": bitstr,
            "opt_config_prob": prob,
            "profit_avg": measure_operator(
                self._model.Hp, self.circuit, self.quantum_instance
            ),
            "violation_avg": measure_operator(
                self._model.Hs, self.circuit, self.quantum_instance
            ),
            "ground_state": x,
        }

        return self.result


class FragmentationSolver(QuantumSolver):
    """
    Generalized fragmented ASP solver.
    """

    def __init__(
        self,
        model: Mine,
        random_seed: Optional[int] = 1953,
        backend=Aer.get_backend("statevector_simulator"),
        shots: Optional[int] = 1024,
    ) -> None:
        super().__init__(model, random_seed, backend, shots)

    def calc_frag_H(self):
        raise NotImplementedError()

    def solve(self) -> Dict:
        raise NotImplementedError()
