import numpy as np
from typing import Union, Optional
from qiskit.aqua.quantum_instance import QuantumInstance
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import PauliOp, WeightedPauliOperator, OperatorBase, LegacyBaseOperator, PauliExpectation, CircuitSampler, StateFn, CircuitStateFn


def identity(n: int, legacy: Optional[bool] = True) -> Union[PauliOp, WeightedPauliOperator]:
    return WeightedPauliOperator([(1.0, Pauli([0]*n, [0]*n))]) if legacy else PauliOp(Pauli([0]*n, [0]*n), coeff=1.0)


def null_operator(n: int, legacy: Optional[bool] = True) -> Union[PauliOp, WeightedPauliOperator]:
    return WeightedPauliOperator([(0.0, Pauli([0]*n, [0]*n))]) if legacy else PauliOp(Pauli([0]*n, [0]*n), coeff=0.0)


def single_qubit_pauli(direction: str, i: int, n: int, legacy: Optional[bool] = True) -> Union[PauliOp, WeightedPauliOperator]:
    zv = [0]*n
    xv = [0]*n
    if(direction == 'z'):
        zv[i] = 1
    if(direction == 'x'):
        xv[i] = 1
    if(direction == 'y'):
        zv[i] = 1
        xv[i] = 1
    zv = np.asarray(zv, dtype=np.bool)
    xv = np.asarray(xv, dtype=np.bool)
    return WeightedPauliOperator([(1.0, Pauli(zv, xv))]) if legacy else PauliOp(Pauli(zv, xv), coeff=0.0)


def z_projector(k: int, i: int, n: int, legacy: Optional[bool] = True) -> Union[PauliOp, WeightedPauliOperator]:
    I = identity(n, legacy=legacy)
    Z = single_qubit_pauli('z', i, n, legacy=legacy)
    return 0.5*I+(-1)**k*0.5*Z

    
def measure_operator(H: Union[OperatorBase, LegacyBaseOperator],
                     circuit: QuantumCircuit,
                     instance: QuantumInstance) -> float:
    if isinstance(H, LegacyBaseOperator):
        H = H.to_opflow()
    return CircuitSampler(instance).convert(PauliExpectation().convert(
        StateFn(H, is_measurement=True).compose(CircuitStateFn(circuit)))).eval().real
