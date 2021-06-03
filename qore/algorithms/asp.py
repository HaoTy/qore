"""The Adiabatic State Preparation algorithm.

See https://arxiv.org/pdf/quant-ph/0001106.pdf
"""

from typing import Optional, Callable, List, Union
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.opflow import OperatorBase, X, I, Minus
from qiskit.algorithms import (
    MinimumEigensolver,
    MinimumEigensolverResult,
    AlgorithmError,
)
from qiskit.utils import QuantumInstance
from qiskit.providers import Backend, BaseBackend

from ..utils import measure_operator


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
            Make sure that :math:`evol_time / nsteps * || H_P - H_B || << 1`,
            where `H_P` is the problem Hamiltonian that will be passed into
            `compute_minimum_eigenvalue()` and `H_B` is the initial Hamiltonian.
        initial_state: qiskit.circuit.QuantumCircuit, optional
            The easily constructable initial state. Should be the ground state
            of the initial_operator H_B.
        initial_operator : qiskit.opflow.OperatorBase, optional
            The initial Hamiltonian H_B. Should have a simple-to-find ground state.
        callback: callable, optional
            A callback function that has access to the partially built circuit
            at each time step. Should accept the circuit (`qiskit.circuit.QuantumCircuit`)
            as the first parameter and the current iteration (`int`) as the
            second parameter.
        callback_freq: int, optional
            The frequency in which the callback function is called.
        """
        self._quantum_instance = None
        self._operator = None
        self._initial_state = None
        self._initial_operator = None
        self._circuit = None
        self._ret = None

        self.quantum_instance = (
            quantum_instance
            if quantum_instance
            else QasmSimulator(method="statevector")
        )
        self.evol_time = evol_time
        self.nsteps = nsteps
        self.initial_state = initial_state
        self.initial_operator = initial_operator
        self._callback = callback
        self._callback_freq = callback_freq

    @property
    def num_qubits(self) -> int:
        for attr in (self._operator, self._initial_operator, self._initial_state):
            if attr is not None:
                return attr.num_qubits
        return 0

    @property
    def initial_state(self) -> Optional[QuantumCircuit]:
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        if initial_state and not self._check_num_qubits(initial_state.num_qubits):
            raise AlgorithmError(
                "The number of qubits of the initial state does not match the "
                "initial operator."
            )
        self._initial_state = initial_state

    @property
    def initial_operator(self) -> Optional[OperatorBase]:
        return self._initial_operator

    @initial_operator.setter
    def initial_operator(self, initial_operator: OperatorBase) -> None:
        if initial_operator and not self._check_num_qubits(initial_operator.num_qubits):
            raise AlgorithmError(
                "The number of qubits of the initial operator does not match "
                "the initial state."
            )
        self._initial_operator = initial_operator

    @property
    def evol_time(self) -> float:
        return self._evol_time

    @evol_time.setter
    def evol_time(self, evol_time: float) -> None:
        self._evol_time = evol_time

    @property
    def nsteps(self) -> int:
        return self._nsteps

    @nsteps.setter
    def nsteps(self, nsteps: int) -> None:
        self._nsteps = nsteps

    @property
    def quantum_instance(self) -> QuantumInstance:
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
        self._initial_state = (Minus ^ self.num_qubits).to_circuit()

    def _set_default_initial_operator(self) -> None:
        self._initial_operator = X ^ (I ^ (self.num_qubits - 1))
        for i in range(1, self.num_qubits):
            self._initial_operator += (I ^ i) ^ X ^ (I ^ (self.num_qubits - i - 1))

    def compute_minimum_eigenvalue(
        self,
        operator: OperatorBase,
        aux_operators: Optional[List[Optional[OperatorBase]]] = None,
    ) -> MinimumEigensolverResult:
        """
        Computes minimum eigenvalue. Operator and aux_operators can be supplied here and
        if not None will override any already set into algorithm so it can be reused with
        different operators. While an operator is required by algorithms, aux_operators
        are optional. To 'remove' a previous aux_operators array use an empty list here.

        Args:
            operator: qiskit.opflow.OperatorBase
                The problem Hamiltonian `H_P`.
            aux_operators: List[qiskit.opflow.OperatorBase], optional
                Optional list of auxiliary operators to be evaluated with the
                eigenstate of the minimum eigenvalue main result and their expectation values
                returned. For instance in chemistry these can be dipole operators, total particle
                count operators so we can get values for these at the ground state.

        Returns:
            MinimumEigensolverResult

        Raises:
            AlgorithmError:
                If no quantum instance has been provided or the number of qubits
                of the operator does not match the initial state or the initial
                operator.
        """
        super().compute_minimum_eigenvalue(operator, aux_operators)

        if self.quantum_instance is None:
            raise AlgorithmError(
                "A QuantumInstance or Backend "
                "must be supplied to run the quantum algorithm."
            )

        if not self._check_num_qubits(operator.num_qubits):
            raise AlgorithmError(
                "The number of qubits of the operator does not match the initial"
                " state or the initial operator."
            )
        self._operator = operator
        self._ret = MinimumEigensolverResult()
        circuit = self._construct_circuit()

        if self._quantum_instance.is_statevector:
            self._ret.eigenstate = self._quantum_instance.execute(
                circuit
            ).get_statevector(circuit)
        else:
            result = self._quantum_instance.execute(circuit.measure_all(inplace=False))
            self._ret.eigenstate = {
                k: (v / self._quantum_instance.run_config.shots) ** 0.5
                for k, v in result.get_counts().items()
            }

        # if aux_operators is not None:
        #     self._ret.aux_operator_eigenvalues = [
        #         measure_operator(op, circuit, self._quantum_instance)
        #         for op in aux_operators
        #     ]
        # self._ret.eigenvalue = measure_operator(
        #     operator, circuit, self._quantum_instance
        # )
        return self._ret
