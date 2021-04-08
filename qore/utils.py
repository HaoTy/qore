import numpy as np
from typing import Union, Optional
from qiskit import QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.opflow import (
    PauliOp,
    OperatorBase,
    PauliExpectation,
    CircuitSampler,
    StateFn,
    CircuitStateFn,
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
) -> float:
    return (
        CircuitSampler(quantum_instance)
        .convert(
            PauliExpectation().convert(
                StateFn(H, is_measurement=True).compose(CircuitStateFn(circuit))
            )
        )
        .eval()
        .real
    )


def get_bitstring_probabilities(circuit: QuantumCircuit, quantum_instance: QuantumInstance) -> dict:
    nqubits = circuit.num_qubits
    qr = QuantumRegister(nqubits, "q")
    cr = ClassicalRegister(nqubits, "c")
    circ = QuantumCircuit(qr, cr)
    for m in range(len(circuit)):
        gate_m = circuit[m][0]
        qubits = [q.index for q in circuit[m][1]]
        circ.append(gate_m, [qr[x] for x in qubits])
    circ.barrier()
    for i in range(circ.num_qubits):
        circ.measure(i, i)
    if quantum_instance.is_statevector:
        quantum_instance = QuantumInstance(Aer.get_backend("qasm_simulator"))
    # else:
    #     backend = quantum_instance.backend
    result = quantum_instance.execute(circ).get_counts()
    return {k: result[k] / quantum_instance.run_config.shots for k in result.keys()}
