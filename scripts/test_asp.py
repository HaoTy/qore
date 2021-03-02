from hamiltonian import *
from adiabatic   import *
from analysis    import *

# how to import a hamiltonian "profit" + "smoothness" part

print("Reading the pit profile...")
Hp,Hs = generate_hamiltonian('4_site_mine.txt',verbose=True)
H     = Hs-Hp
print("Hamiltonian operator...")
print(H.print_details())

# how to run a ASP calculation

from qiskit                                        import *
from qiskit.aqua                                   import aqua_globals
from qiskit.aqua.algorithms                        import QAOA
from qiskit.aqua.components.optimizers             import COBYLA
from qiskit.circuit.library                        import TwoLocal
from qiskit.optimization.applications.ising.common import sample_most_likely

aqua_globals.random_seed = 1953
backend                  = Aer.get_backend('statevector_simulator')
instance                 = QuantumInstance(backend=backend)
asp_circuit              = ASP(H,T=15.0,nsteps=8)

print(asp_circuit.draw())
asp_analysis = analysis('4_site_mine.txt',asp_circuit,instance)

print("Average Profit:                     ",asp_analysis['profit'])
print("Average # of Smoothness Violations: ",asp_analysis['smoothness_violations'])
print("Bitstrings and their probabilities: ")
for k in asp_analysis['bitstring_p'].keys():
    print(k,asp_analysis['bitstring_p'][k])

