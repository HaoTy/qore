import numpy as np

from qiskit.quantum_info                           import Pauli
from qiskit.aqua.operators                         import WeightedPauliOperator
from qiskit.aqua.components.variational_forms      import VariationalForm
from qiskit.aqua                                   import QuantumInstance,aqua_globals
from qiskit.providers.aer.noise                    import NoiseModel
from qiskit.compiler                               import transpile
from qiskit.circuit.library.n_local                import EfficientSU2
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.aqua.components.optimizers             import CG,L_BFGS_B,COBYLA
from qiskit.aqua.algorithms                        import VQE
from qiskit.aqua.components.initial_states         import InitialState
from qiskit.aqua.operators                         import Z2Symmetries

# ========================================================================== #

def identity(n):
    zv = [0]*n
    xv = [0]*n
    return WeightedPauliOperator([(1.0,Pauli(zv,xv))])

def null_operator(n):
    zv = [0]*n
    xv = [0]*n
    return WeightedPauliOperator([(0.0,Pauli(zv,xv))])

def single_qubit_pauli(direction,i,n):
    zv = [0]*n
    xv = [0]*n
    if(direction=='z'): zv[i] = 1
    if(direction=='x'): xv[i] = 1
    if(direction=='y'): zv[i] = 1; xv[i] = 1
    zv = np.asarray(zv,dtype=np.bool)
    xv = np.asarray(xv,dtype=np.bool)
    return WeightedPauliOperator([(1.0,Pauli(zv,xv))])

def z_projector(k,i,n):
    I = identity(n)
    Z = single_qubit_pauli('z',i,n)
    return 0.5*I+(-1)**k*0.5*Z

def diagonalize(H,verbose=False):
    from qiskit.aqua.algorithms import NumPyEigensolver
    ee       = NumPyEigensolver(H,k=2**H.num_qubits)
    res      = np.real(ee.run()['eigenvalues'])
    spectrum = set([round(x,6) for x in res])
    for s in sorted(spectrum):
        idx_s = [r for r in res if np.abs(r-s)<1e-6]
        if(verbose): print("eigenvalue, degeneracy ",s,len(idx_s))
    return sorted(spectrum)



