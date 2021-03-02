"""The Adiabatic State Preparation algorithm.

See https://arxiv.org/pdf/quant-ph/0001106.pdf
"""

from typing import Optional, Union, Dict
from qiskit.aqua.algorithms import AlgorithmResult, QuantumAlgorithm
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.aqua.components.initial_states import InitialState
from qiskit.providers import BaseBackend, Backend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import OperatorBase, LegacyBaseOperator
from qiskit.aqua import AquaError
# import numpy as np
from qore.utils import single_qubit_pauli, null_operator


class ASP(QuantumAlgorithm):
    """
    The Adiabatic State Preparation algorithm.

    See https://arxiv.org/pdf/quant-ph/0001106.pdf
    """

    def __init__(self,
                 H_P: Union[OperatorBase, LegacyBaseOperator],
                 T: Union(float, int),
                 nsteps: int,
                 initial_state: Optional[QuantumCircuit] = None,
                 H_B: Optional[Union[OperatorBase, LegacyBaseOperator]] = None,
                 quantum_instance: Optional[
            Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        super().__init__(quantum_instance)
        self.H_P = H_P
        self.evol_time = T
        self.nsteps = nsteps
        self.initial_state = initial_state.copy() or StandardASPInitialState(
            H_P.num_qubits)
        self.H_B = H_B.copy() or construct_default_H_B(H_P.num_qubits)

    @property
    def num_qubits(self) -> int:
        return self.H_P.num_qubits

    def construct_circuit(self) -> QuantumCircuit:
        circ = self.initial_state
        for i in range(self.nsteps):
            xi = i / self.nsteps
            circ = ((1 - xi) * self.H_B + xi * self.self.H_P).evolve(
                circ, evo_time=self.evol_time / self.nsteps)
        self.circuit = circ
        return circ

    def _run(self) -> Dict:
        return {'circuit': self.construct_circuit()}


class StandardASPInitialState(InitialState):
    def __init__(self, num_qubits: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits

    def construct_circuit(self,
                          mode: str = 'circuit',
                          register: Optional[QuantumRegister] = None) -> QuantumCircuit:
        if mode == 'circuit':
            circ = QuantumCircuit(
                register or QuantumRegister(self.num_qubits, 'q'))
            for i in range(self.num_qubits):
                circ.h(i)
            return circ
        elif mode == 'vector':
            raise NotImplementedError()
        else:
            raise AquaError(
                'Parameter `mode` needs to be either "circuit" or "vector"')


def construct_default_H_B(num_qubits: int) -> Union[OperatorBase, LegacyBaseOperator]:
    H_B = null_operator(num_qubits)
    for i in range(num_qubits):
        H_B += single_qubit_pauli('x', i, num_qubits)
    return H_B
