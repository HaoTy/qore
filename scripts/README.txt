4_site_mine.txt: example of a pit profile (see hamiltonian.py for its processing)
test_asp.py:     adiabatic state preparation (ATTENTION: not native qiskit, needs debug)
test_qaoa.py:    qaoa calculation (https://qiskit.org/documentation/stubs/qiskit.aqua.algorithms.QAOA.html)
test_vqe.py:     vqe calculation using a "hardware efficient Ansatz"

hamiltonian.py:  read a file describing a pit profile and construct the corresponding Hamiltonian
adiabatic.py:    implementation of the adiabatic state preparation algorithm
analysis.py:     post-processing, compute the expectation value of the Hamiltonian, 
                                  assess if there are smoothness violations,
                                  get the probability distribution for the bitstrings

