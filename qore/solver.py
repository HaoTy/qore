from typing import Optional, Union, Dict, List
import numpy as np

from qiskit.utils import QuantumInstance
from qiskit.providers import Backend, BaseBackend
from qiskit.providers.aer import QasmSimulator
from qiskit.algorithms import MinimumEigensolver
from qiskit.opflow import Minus

from .model.mine import Mine, SubMine, MiningProblemResult
from .utils import measure_operator


class FragmentationSolver():
    """
    Open-pit mining problem solver with fragmentation.
    """

    def __init__(self,
                 qmine: Mine,
                 node_lists: List,
                 quantum_instance: Optional[Union[QuantumInstance,
                                                  BaseBackend, Backend]] = None,
                 shots: Optional[int] = 1024,
                 ) -> None:
        # quantum backends
        if quantum_instance:
            self.quantum_instance = quantum_instance
        else:
            backend = QasmSimulator(method='statevector')
            self.quantum_instance = QuantumInstance(backend)
        self.shots = shots

        # submines
        self.qmine = qmine
        self.submines = []  # submine objects
        self.z_val = {}
        self._init_submine(node_lists)

        # energy
        self.nfrags = len(self.submines)
        self.expected_profit = [0.]*self.nfrags
        self.expected_violation = [0.]*self.nfrags
        self._init_energy()  # TODO: random initialization

        # optimal configuration
        self.optimal_config = ['0']*self.qmine.nqubits

    def _init_submine(self, node_lists: List) -> None:
        for node_list in node_lists:
            submine = SubMine(self.qmine, node_list)
            self.submines.append(submine)
            for i in submine.node_c:
                self.z_val[i] = 0

    def _init_energy(self) -> None:
        for idx, submine in enumerate(self.submines):
            init_state = (Minus ^ submine.nqubits).to_circuit()
            self.expected_profit[idx] = measure_operator(
                submine.Hp, init_state, self.quantum_instance)
            self.expected_violation[idx] = measure_operator(
                submine.Hs, init_state, self.quantum_instance)

    def cal_energy(self, penalty: float) -> float:
        res_bd = 0.0
        for submine in self.submines:
            for p in submine.node_c:
                for c in submine.node_c[p]:
                    res_bd += 0.5*(1-self.z_val[p])*0.5*(1+self.z_val[c])

        return sum(self.expected_profit) + penalty*(sum(self.expected_violation) + res_bd)

    def get_config(self) -> str:
        return "".join(self.optimal_config[::-1])

    def calc(
        self,
        algorithm: MinimumEigensolver,
        penalty: float,
        tol: Optional[float] = 1e-5
    ) -> "MiningProblemResult":
        energy = sum(self.expected_profit) + penalty * \
            sum(self.expected_violation)
        del_e = tol+0.1
        count = 0
        while del_e > tol:
            for idx, submine in enumerate(self.submines):
                print(f"calculating submine {idx}")
                res = submine.solve(algorithm, penalty, self.z_val)

                self.expected_profit[idx] = res.expected_profit
                self.expected_violation[idx] = res.expected_violation

                optimal_config = list(res.optimal_config)[::-1]
                for idx, b in zip(submine.node_list, optimal_config):
                    self.optimal_config[idx] = b

            energy_new = self.cal_energy(penalty)
            energy, del_e = energy_new, abs(energy_new - energy)

            count += 1
            print(f"iteration {count}")
            print(f"del_e={del_e}")
            print("---------------------------------------------")

        self._ret = MiningProblemResult()
        self._ret.optimal_config = self.get_config()
        self._ret.expected_profit = sum(self.expected_profit)
        self._ret.expected_violation = (
            energy - self._ret.expected_profit) / penalty
        return self._ret
