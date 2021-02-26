from hamiltonian import *
from analysis    import *

# how to import a hamiltonian "profit" + "smoothness" part

print("Reading the pit profile...")
Hp,Hs = generate_hamiltonian('4_site_mine.txt',verbose=True)
H     = Hs-Hp
print("Hamiltonian operator...")
print(H.print_details())

# how to run a VQE calculation

from qiskit                                        import *
from qiskit.aqua                                   import aqua_globals
from qiskit.aqua.algorithms                        import QAOA
from qiskit.aqua.components.optimizers             import COBYLA
from qiskit.circuit.library                        import TwoLocal
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.circuit.library.n_local                import EfficientSU2
from qiskit.aqua.algorithms                        import VQE

# starting from a uniform superposition of all bitstrings

aqua_globals.random_seed = 1953
backend                  = Aer.get_backend('statevector_simulator')
instance                 = QuantumInstance(backend=backend)
optimizer                = COBYLA()
var_form                 = EfficientSU2(num_qubits=H.num_qubits,reps=3,su2_gates=['ry'],entanglement='linear',
                                        skip_unentangled_qubits=False,skip_final_rotation_layer=False,parameter_prefix='t',insert_barriers=True)
algo                     = VQE(H,var_form,optimizer)
algo_result              = algo.run(instance)
algo_circuit             = algo.get_optimal_circuit()

print(algo_circuit.draw())
algo_analysis = analysis('4_site_mine.txt',algo_circuit,instance)

print("Average Profit:                     ",algo_analysis['profit'])
print("Average # of Smoothness Violations: ",algo_analysis['smoothness_violations'])
print("Bitstrings and their probabilities: ")
for k in algo_analysis['bitstring_p'].keys():
    print(k,algo_analysis['bitstring_p'][k])

