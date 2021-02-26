from qiskit import *
from hamiltonian import *

def initial_circuit(n):
    M = null_operator(n)
    for i in range(n):
        M += single_qubit_pauli('x',i,n)
    qr      = QuantumRegister(n,'q')
    cr      = ClassicalRegister(n,'c')
    circ    = QuantumCircuit(qr,cr)
    for i in range(n):
        circ.h(i)
    return M,circ

def ASP(H,T,nsteps):
    M,circ = initial_circuit(H.num_qubits)
    for i in range(nsteps):
        xi  = float(i)/float(nsteps)
        yi  = (1.0-xi)
        circ = (yi*M+xi*H).evolve(circ,evo_time=T/float(nsteps),num_time_slices=1,expansion_mode='trotter',expansion_order=1)
    return circ.copy()



