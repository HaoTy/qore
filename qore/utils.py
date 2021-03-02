import numpy as np
from typing import Union, Optional
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import PauliOp, WeightedPauliOperator


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
