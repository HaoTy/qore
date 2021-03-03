from hamiltonian import *
from adiabatic   import *
from analysis    import *

# how to import a hamiltonian "profit" + "smoothness" part

print("Reading the pit profile...")
Hp,Hs = generate_hamiltonian('4_site_mine.txt',verbose=True)
H     = Hs-Hp
print("Hamiltonian operator...")
print(H.print_details())

# how to run an ASP calculation

from qiskit                                        import *
from qiskit.aqua                                   import aqua_globals
from qiskit.aqua.algorithms                        import QAOA
from qiskit.aqua.components.optimizers             import COBYLA
from qiskit.circuit.library                        import TwoLocal
from qiskit.optimization.applications.ising.common import sample_most_likely

T_list  = [1.0,2.0,5.0,10.0,15.0,20.0]
nt_list = [10*x for x in  range(2,21)] 
results = np.zeros((len(T_list),len(nt_list),2))

for jT,T in enumerate(T_list):
    for jn,nt in enumerate(nt_list):

        aqua_globals.random_seed = 1953
        backend                  = Aer.get_backend('statevector_simulator')
        instance                 = QuantumInstance(backend=backend)
        asp_circuit              = ASP(H,T=T,nsteps=nt)

        # instead of looking at those awful "evolution" blocks, let us transpile the circuit
        from qiskit.compiler import transpile
        asp_circuit = transpile(asp_circuit,backend,optimization_level=0)
        if(jT==0 and jn==0): print(asp_circuit.draw())
        
        asp_analysis = analysis('4_site_mine.txt',asp_circuit,instance)
        results[jT,jn,0] = asp_analysis['profit'][0]
        results[jT,jn,1] = asp_analysis['smoothness_violations'][0]
        
        #print("Average Profit:                     ",asp_analysis['profit'])
        #print("Average # of Smoothness Violations: ",asp_analysis['smoothness_violations'])
        #print("Bitstrings and their probabilities: ")
        #for k in asp_analysis['bitstring_p'].keys():
        #   print(k,asp_analysis['bitstring_p'][k])

import matplotlib.pyplot as plt

fig,axs = plt.subplots(1,2,figsize=(10,5))

for jT,T in enumerate(T_list):
    axs[0].plot([1.0/float(x) for x in nt_list],results[jT,:,0],label="T = "+str(T))
    axs[1].plot([1.0/float(x) for x in nt_list],results[jT,:,1],label="T = "+str(T))

axs[0].set_ylabel('profit [arb. units]')
axs[1].set_ylabel('smoothness violation [gamma]')
axs[0].set_xlabel('1/number of time steps')
axs[1].set_xlabel('1/number of time steps')
axs[0].legend()
axs[1].legend()
plt.show()
