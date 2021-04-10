"""The Adiabatic State Preparation algorithm.

See https://arxiv.org/pdf/quant-ph/0001106.pdf
"""

from typing import Optional, Dict, Callable, List, Union
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.opflow import OperatorBase
from qiskit.algorithms import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    AlgorithmError,
)
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend, BaseBackend

from qore.utils import get_bitstring_probabilities, single_qubit_pauli, null_operator


class ASP(MinimumEigensolver):
    """
    The Adiabatic State Preparation algorithm.

    See https://arxiv.org/pdf/quant-ph/0001106.pdf
    """

    def __init__(
        self,
        evol_time: float,
        nsteps: int,
        initial_state: Optional[QuantumCircuit] = None,
        initial_operator: Optional[OperatorBase] = None,
        # expectation: Optional[ExpectationBase] = None,
        # include_custom: bool = False,
        callback: Optional[Callable[[QuantumCircuit, int], None]] = None,
        callback_freq: Optional[int] = 1,
        quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
    ) -> None:
        """
        Initialize the Adiabatic State Preparation algorithm.

        Parameters
        ----------
        evol_time: float
            Time for the adiabatic evolution.
        nsteps: int
            Number of discrete time blocks for Trotterization.
            Make sure that evol_time / nsteps * || H_P - H_B || << 1.
        initial_state: qiskit.circuit.QuantumCircuit, optional
            The easily constructable initial state. Should be the ground state of H_B.
        initial_operator : qiskit.opflow.OperatorBase, optional
            The initial Hamiltonian. Should have a simple-to-find ground state.
        callback: callable, optional
            A callback function that has access to the partially built circuit at each time step.
        callback_freq: int, optional
            The frequency in which the callback function is called.
        """
        self._quantum_instance = None
        self._operator = None
        self._initial_state = None
        self._initial_operator = None
        self._circuit = None
        self._ret = None

        if quantum_instance:
            self.quantum_instance = quantum_instance
        self.evol_time = evol_time
        self.nsteps = nsteps
        self.initial_state = initial_state
        self.initial_operator = initial_operator
        # self.initial_state = (
        #     StandardASPInitialState(operator.num_qubits).construct_circuit()
        #     if initial_state is None
        #     else initial_state.copy()
        # )
        # self.initial_operator = construct_default_H_B(operator.num_qubits) if initial_operator is None else initial_operator.copy()
        self._callback = callback
        self._callback_freq = callback_freq

    @property
    def num_qubits(self) -> int:
        for op in (self._operator, self._initial_operator, self._initial_state):
            if op is not None:
                return op.num_qubits
        return 0

    @property
    def initial_state(self) -> QuantumCircuit:
        return self._evol_time

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        if initial_state and not self._check_num_qubits(initial_state.num_qubits):
            raise AlgorithmError(
                "The number of qubits of the initial state does not match the initial operator."
            )
        self._initial_state = initial_state

    @property
    def initial_operator(self) -> OperatorBase:
        return self._initial_operator

    @initial_operator.setter
    def initial_operator(self, initial_operator: OperatorBase) -> None:
        if initial_operator and not self._check_num_qubits(initial_operator.num_qubits):
            raise AlgorithmError(
                "The number of qubits of the initial operator does not match the initial state."
            )
        self._initial_operator = initial_operator

    # @property
    # def evol_time(self) -> float:
    #     return self._evol_time

    # @evol_time.setter
    # def evol_time(self, evol_time: float) -> None:
    #     self._evol_time = evol_time

    # @property
    # def nsteps(self) -> int:
    #     return self._nsteps

    # @nsteps.setter
    # def nsteps(self, nsteps: int) -> None:
    #     self._nsteps = nsteps

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

    def _construct_circuit(self) -> QuantumCircuit:
        if self._initial_state is None:
            self._set_default_initial_state()
        if self._initial_operator is None:
            self._set_default_initial_operator()
        circuit = self._initial_state.copy()
        for i in range(self.nsteps):
            xi = (0.5 + i) / self.nsteps
            circuit.hamiltonian(
                self._operator,
                xi * self.evol_time / self.nsteps,
                list(range(self.num_qubits)),
            )
            circuit.hamiltonian(
                self._initial_operator,
                (1 - xi) * self.evol_time / self.nsteps,
                list(range(self.num_qubits)),
            )
            if self._callback and i % self._callback_freq == 0:
                self._callback(circuit, i)

        self._circuit = circuit
        return circuit

    def _check_num_qubits(self, num_qubits: int) -> bool:
        return self.num_qubits == 0 or self.num_qubits == num_qubits

    def _set_default_initial_state(self) -> None:
        self._initial_state = QuantumCircuit(QuantumRegister(self.num_qubits, "q"))
        for i in range(self.num_qubits):
            self._initial_state.x(i)
            self._initial_state.h(i)

    def _set_default_initial_operator(self) -> None:
        self._initial_operator = null_operator(self.num_qubits)
        for i in range(self.num_qubits):
            self._initial_operator += single_qubit_pauli("x", i, self.num_qubits)

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        aux_operators: Optional[List[Optional[OperatorBase]]] = None,
    ) -> MinimumEigensolverResult:
        super().compute_minimum_eigenvalue(operator, aux_operators)

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend "
                "must be supplied to run the quantum algorithm."
            )

        if not self._check_num_qubits(operator.num_qubits):
            raise AlgorithmError(
                "The number of qubits of the operator does not match the initial state or the initial operator."
            )
        self._operator = operator
        self._ret = MinimumEigensolverResult()
        self._ret.eigenstate = get_bitstring_probabilities(
            self._construct_circuit(), self.quantum_instance
        )
        # self._ret.eigenvalue =  TODO: implement expectation construction and eigenvalue calculation
        return self._ret
