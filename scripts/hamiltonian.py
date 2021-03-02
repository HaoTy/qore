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

# ========================================================================== #

def read_info(fname,verbose=False):
    # read info about H in a file, of the form
    # number of sites
    # penalty
    # i x y profit(i) parents
    # and dump this info in a dictionary
    h_dict = {'n'           : 0,
              'penalty'     : 0,
              'coordinates' : [],
              'profit'      : [],
              'parents'     : []}
    infile = open(fname,'r')
    infile = infile.readlines()
    h_dict['n']       = int(infile[0])
    h_dict['penalty'] = float(infile[1])
    for l in infile[2:]:
        l   = l.split()
        x,y = int(l[0]),int(l[1])
        w   = float(l[2])
        h_dict['coordinates'].append((x,y))
        h_dict['profit'].append(w)
        if(len(l)==4): h_dict['parents'].append([])
        else:          h_dict['parents'].append([int(x) for x in l[3:]])
    # plot the matrix of profit
    if(verbose):
       nr = max([x for (x,y) in h_dict['coordinates']])+1
       nc = max([y for (x,y) in h_dict['coordinates']])+1
       w = np.zeros((nr,nc))
       for i,(x,y) in enumerate(h_dict['coordinates']):
           w[x,y] = h_dict['profit'][i]
       from prettytable import PrettyTable
       x = PrettyTable([' ']+[str(ic) for ic in range(nc)])
       for ir in range(nr):
           x.add_row([ir]+['%.3f' % w[ir,ic] for ic in range(nc)])
       print(str(x))
    return h_dict

def generate_hamiltonian(fname,verbose=False):
    h_dict = read_info(fname,verbose)
    # H profit: \sum_i w(i) (1-Zi)/2
    Hp = null_operator(h_dict['n'])
    for i in range(h_dict['n']):
        Hp += h_dict['profit'][i]*z_projector(1,i,h_dict['n'])
    # H smoothness: gamma * \sum_i \sum_{j parent of i} (1-Zi)/2 * (1+Zj)/2
    Hs = null_operator(h_dict['n'])
    for i in range(h_dict['n']):
        for j in h_dict['parents'][i]:
            Hs += h_dict['penalty']*z_projector(1,i,h_dict['n'])*z_projector(0,j,h_dict['n'])
    return Hp,Hs


