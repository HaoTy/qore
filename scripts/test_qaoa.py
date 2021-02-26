from hamiltonian import *
from analysis    import *

# how to import a hamiltonian "profit" + "smoothness" part

print("Reading the pit profile...")
Hp,Hs = generate_hamiltonian('4_site_mine.txt',verbose=True)
H     = Hs-Hp
print("Hamiltonian operator...")
print(H.print_details())

# how to run a QAOA calculation

from qiskit                                        import *
from qiskit.aqua                                   import aqua_globals
from qiskit.aqua.algorithms                        import QAOA
from qiskit.aqua.components.optimizers             import COBYLA
from qiskit.circuit.library                        import TwoLocal
from qiskit.optimization.applications.ising.common import sample_most_likely

aqua_globals.random_seed = 1953
backend                  = Aer.get_backend('statevector_simulator')
instance                 = QuantumInstance(backend=backend)
optimizer                = COBYLA()
qaoa                     = QAOA(H,optimizer,quantum_instance=instance,p=5)
qaoa_result              = qaoa.run()
qaoa_circuit             = qaoa.get_optimal_circuit()

print(qaoa_circuit.draw())
qaoa_analysis = analysis('4_site_mine.txt',qaoa_circuit,instance)

print("Average Profit:                     ",qaoa_analysis['profit'])
print("Average # of Smoothness Violations: ",qaoa_analysis['smoothness_violations'])
print("Bitstrings and their probabilities: ")
for k in qaoa_analysis['bitstring_p'].keys():
    print(k,qaoa_analysis['bitstring_p'][k])

