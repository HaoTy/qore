import numpy as np
from typing import Optional, Union
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.providers import Backend, BaseBackend, aer
from qiskit.opflow import (
    PauliOp,
    OperatorBase,
    CircuitSampler,
    StateFn,
    CircuitStateFn,
    ExpectationBase,
    ExpectationFactory,
)


def identity(n: int) -> PauliOp:
    return PauliOp(Pauli(([0] * n, [0] * n)), coeff=1.0)


def null_operator(n: int) -> PauliOp:
    return PauliOp(Pauli(([0] * n, [0] * n)), coeff=0.0)


def single_qubit_pauli(direction: str, i: int, n: int) -> PauliOp:
    zv = [0] * n
    xv = [0] * n
    if direction == "z":
        zv[i] = 1
    if direction == "x":
        xv[i] = 1
    if direction == "y":
        zv[i] = 1
        xv[i] = 1
    zv = np.asarray(zv, dtype=np.bool)
    xv = np.asarray(xv, dtype=np.bool)
    return PauliOp(Pauli((zv, xv)), coeff=1.0)


def z_projector(k: int, i: int, n: int) -> PauliOp:
    return 0.5 * identity(n) + (-1) ** k * 0.5 * single_qubit_pauli("z", i, n)


def measure_operator(
    H: OperatorBase,
    circuit: QuantumCircuit,
    quantum_instance: QuantumInstance,
    expectation: Optional[ExpectationBase] = None,
) -> float:

    if expectation is None:
        expectation = ExpectationFactory.build(H, quantum_instance)

    return (
        CircuitSampler(quantum_instance)
        .convert(
            expectation.convert(
                StateFn(H, is_measurement=True).compose(CircuitStateFn(circuit))
            )
        )
        .eval()
        .real
    )

def int_to_bitstr(x: int, n: int):
    return format(x, f"0{n + 2}b")[2:]


def get_bitstring_probabilities(
    circuit: QuantumCircuit,
    quantum_instance: Optional[Union[QuantumInstance, BaseBackend, Backend]] = None,
) -> dict:

    if quantum_instance is None:
        quantum_instance = aer.QasmSimulator(method="statevector")

    if not isinstance(quantum_instance, QuantumInstance):
        quantum_instance = QuantumInstance(quantum_instance)

    result = quantum_instance.execute(circuit.measure_all(inplace=False))

    if quantum_instance.is_statevector:
        statevector = result.get_statevector(circuit)
        return {
            int_to_bitstr(k, circuit.num_qubits): v
            for k, v in enumerate(statevector * statevector.conj())
        }

    return {
        k: v / quantum_instance.run_config.shots
        for (k, v) in result.get_counts().items()
    }
