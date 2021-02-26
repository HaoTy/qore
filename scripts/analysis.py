import numpy as np
import functools
from qiskit import *
from hamiltonian import *

def measure_operator(H,circuit,instance):
    circuits = H.construct_evaluation_circuit(
              wave_function               = circuit,
              statevector_mode            = instance.is_statevector,
              use_simulator_snapshot_mode = instance.is_statevector,
              circuit_name_prefix         = 'H')

    to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
    result = instance.execute(to_be_simulated_circuits)

    res = np.zeros(2)
    mean,std = H.evaluate_with_result(
               result                      = result,
               statevector_mode            = instance.is_statevector,
               use_simulator_snapshot_mode = instance.is_statevector,
               circuit_name_prefix         = 'H')
    return np.real(mean),np.abs(std)

def get_bitstring_probabilities(circuit,instance,shots):
    nqubits = circuit.num_qubits
    qr      = QuantumRegister(nqubits,'q')
    cr      = ClassicalRegister(nqubits,'c')
    circ    = QuantumCircuit(qr,cr)
    for m in range(len(circuit)):
        gate_m = circuit[m][0]
        qubits = [q.index for q in circuit[m][1]]
        circ.append(gate_m,[qr[x] for x in qubits])
    circ.barrier()
    for i in range(circ.num_qubits): circ.measure(i,i)
    if(instance.is_statevector): backend = Aer.get_backend('qasm_simulator')
    else:                        backend = instance.backend
    print(circ)
    result = execute(circ,backend=backend,shots=shots).result().get_counts()
    return {k:result[k]/float(shots) for k in result.keys()}

def analysis(fname,circuit,instance,shots=1024):
    h_dict = read_info(fname)
    Hp,Hs = generate_hamiltonian(fname)
    res_p = measure_operator(Hp,circuit,instance)
    res_s = measure_operator(Hs,circuit,instance)
    x     = get_bitstring_probabilities(circuit,instance,shots)
    g     = h_dict['penalty']
    return {'profit'                : (res_p[0],res_p[1]),
            'smoothness_violations' : (res_s[0]/g,res_s[1]/g),
            'bitstring_p'           : x}

