"""The Adiabatic State Preparation algorithm.

See https://arxiv.org/pdf/quant-ph/0001106.pdf
"""

from typing import Optional, Union, Dict, List, Callable
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qiskit.providers import BaseBackend, Backend
from qiskit.utils import QuantumInstance
from qiskit.opflow import OperatorBase, PauliTrotterEvolution, StateFn
from qiskit.extensions import HamiltonianGate

# import numpy as np
from qore.utils import single_qubit_pauli, null_operator


class ASP:
    """
    The Adiabatic State Preparation algorithm.

    See https://arxiv.org/pdf/quant-ph/0001106.pdf
    """

    def __init__(
        self,
        H_P: OperatorBase,
        evol_time: float,
        nsteps: int,
        initial_state: Optional[QuantumCircuit] = None,
        H_B: Optional[OperatorBase] = None,
        callback: Optional[Callable[[QuantumCircuit], None]] = None,
        callback_freq: Optional[int] = 1,
        # quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Initialize the Adiabatic State Preparation algorithm.

        Parameters
        ----------
        H_P : qiskit.opflow.OperatorBase
            The problem Hamiltonian.
        evol_time: float
            Time for the adiabatic evolution.
        nsteps: int
            Number of discrete time blocks for Trotterization.
            Make sure that evol_time / nsteps * || H_P - H_B || << 1.
        initial_state: qiskit.circuit.QuantumCircuit, optional
            The easily constructable initial state. Should be the ground state of H_B.
        H_B : qiskit.opflow.OperatorBase, optional
            The initial Hamiltonian. Should have a simple-to-find ground state.
        callback: callable, optional
            A callback function that has access to the partially built circuit at each time step.
        callback_freq: int, optional
            The frequency in which the callback function is called.
        """
        self.H_P = H_P
        self.evol_time = evol_time
        self.nsteps = nsteps
        self.initial_state = (
            StandardASPInitialState(H_P.num_qubits).construct_circuit()
            if initial_state is None
            else initial_state.copy()
        )
        self.H_B = construct_default_H_B(H_P.num_qubits) if H_B is None else H_B.copy()
        self._callback = callback
        self._freq = callback_freq

    @property
    def num_qubits(self) -> int:
        return self.H_P.num_qubits

    def construct_circuit(self) -> QuantumCircuit:
        circ = self.initial_state
        for i in range(self.nsteps):
            xi = (0.5 + i) / self.nsteps
            circ.hamiltonian(self.H_P, xi * self.evol_time / self.nsteps, list(range(self.num_qubits)))
            circ.hamiltonian(self.H_B, (1 - xi) * self.evol_time / self.nsteps, list(range(self.num_qubits)))
            if self._callback and i % self._freq == 0:
                self._callback(circ)

        self.circuit = circ
        return circ

    def run(self) -> Dict:
        return {"circuit": self.construct_circuit()}


class StandardASPInitialState:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def construct_circuit(
        self, mode: str = "circuit", register: Optional[QuantumRegister] = None
    ) -> QuantumCircuit:
        if mode == "circuit":
            circ = QuantumCircuit(register or QuantumRegister(self.num_qubits, "q"))
            for i in range(self.num_qubits):
                circ.x(i)
                circ.h(i)
            return circ
        elif mode == "vector":
            raise NotImplementedError()
        else:
            raise ValueError(
                'Parameter `mode` needs to be either "circuit" or "vector"'
            )


def construct_default_H_B(num_qubits: int) -> OperatorBase:
    H_B = null_operator(num_qubits)
    for i in range(num_qubits):
        H_B += single_qubit_pauli("x", i, num_qubits)
    return H_B
