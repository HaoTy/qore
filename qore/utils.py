import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator


def identity(n):
    zv = [0]*n
    xv = [0]*n
    return WeightedPauliOperator([(1.0, Pauli(zv, xv))])


def null_operator(n):
    zv = [0]*n
    xv = [0]*n
    return WeightedPauliOperator([(0.0, Pauli(zv, xv))])


def single_qubit_pauli(direction, i, n):
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
    return WeightedPauliOperator([(1.0, Pauli(zv, xv))])
