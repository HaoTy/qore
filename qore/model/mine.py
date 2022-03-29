"""The Mine class."""

__docformat__ = "reStructuredText"

from functools import reduce
from typing import List, Dict, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from prettytable import PrettyTable
from qiskit.opflow import PauliOp, I, Z, Plus, MatrixOp
from qiskit.algorithms import (
    MinimumEigensolver,
    AlgorithmResult,
    NumPyMinimumEigensolver,
)

from ..algorithms import Pseudoflow, PEPSITE
from ..benchmark import Benchmark
from ..utils import null_operator, z_projector, single_qubit_pauli, int_to_bitstr


class BaseMine(ABC):
    """
    Abstract Mine class
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def solve(self) -> "MiningProblemResult":
        raise NotImplementedError()


class Mine(BaseMine):
    """This class stores the mine configurations."""

    def __init__(self, mine_config: Union[str, np.ndarray]) -> None:
        """Initialize the Mine class.

        Parameters
        ----------
        mine_config : str
            Path to the mine configuration file.

        Raises
        ------
        IOError
            Invalid path to the mine configuration file.

        """
        if isinstance(mine_config, str):
            try:
                self.dat = np.loadtxt(mine_config, dtype=float)
            except:
                raise IOError("Invalid Mine Configuration File")
        elif isinstance(mine_config, np.ndarray):
            if len(mine_config.shape) != 2:
                raise ValueError("`mine_config` must be two-demensional")
            self.dat = np.array(mine_config, dtype=float)
        else:
            raise ValueError("Unrecognized `mine_config` type")

        self.rows, self.cols = self.dat.shape
        self.graph = defaultdict(list)  # p:c
        self.graph_r = defaultdict(list)  # c:p
        self.idx2cord = []
        self.cord2idx = {}
        self._init_mapping()
        self.nqubits: int = len(self.idx2cord)
        self.valid_configs = None
        self._Hs = None
        self._Hp = None

    @property
    def Hs(self) -> float:
        if self._Hs is None:
            self._Hs = self.gen_Hs()
        return self._Hs

    @property
    def Hp(self) -> float:
        if self._Hp is None:
            self._Hp = self.gen_Hp()
        return self._Hp

    @staticmethod
    def gen_random_mine(
        size: Tuple[int, int], distribution: Optional[str] = "normal", seed: Optional[int] = None, **kwargs
    ) -> "Mine":
        distribution = distribution.lower()
        rng = np.random.default_rng(seed)
        if distribution == "gaussian" or distribution == "normal":
            return Mine(rng.normal(size=size, **kwargs))
        if distribution == "uniform":
            return Mine(rng.uniform(size=size, **kwargs))
        raise ValueError("Distribution not supported.")

    def _init_mapping(self) -> None:
        """Assign a unique id to each valid node and store the graph structure with ``self.graph``."""
        for r in range(self.rows):
            for c in range(self.cols):
                if c < r or c > self.cols - r - 1:
                    self.dat[r, c] = float("inf")
                if self.dat[r, c] < float("inf"):
                    self.idx2cord.append((r, c))
                    self.cord2idx[(r, c)] = len(self.idx2cord) - 1
                    idx = self.cord2idx[(r, c)]
                    for pr, pc in [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1)]:
                        if (
                            0 <= pr < self.rows
                            and 0 <= pc < self.cols
                            and self.dat[pr, pc] < float("inf")
                        ):
                            self.graph[idx].append(self.cord2idx[(pr, pc)])
                            self.graph_r[self.cord2idx[(pr, pc)]].append(idx)

    def plot_mine(self) -> None:
        """Plot the mine configuration."""
        x = PrettyTable([" "] + [str(ic) for ic in range(self.cols)])
        for ir in range(self.rows):
            x.add_row([ir] + ["%.3f" % self.dat[ir, ic] for ic in range(self.cols)])
        print(str(x))

    def plot_mine_graph(
        self, color: str = "r", pos_func: Callable = nx.spring_layout
    ) -> None:
        """Plot a graph representing the mine configuration.

        Parameters
        ----------
        color : str
            Color of the nodes.
        pos_func : Callable
            A Callable that returns positions of nodes.

        """
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.nqubits))
        elist = [[i, j] for i, jl in self.graph.items() for j in jl]
        # tuple is (i,j,weight) where (i,j) is the edge
        G.add_edges_from(elist)

        colors = [color for node in G.nodes()]
        pos = pos_func(G)

        default_axes = plt.axes(frameon=True)
        nx.draw_networkx(
            G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos
        )

    def gen_Hs(self) -> PauliOp:
        """Generate the smoothess Hamiltonian
        :math:`H_{s}=\sum_{i}\sum_{j:Parent(i)} (1-Z_{i})/2*(1+Z_{j})/2`

        Returns
        ----------
        qiskit.opflow.PauliOp
            Smoothness Hamiltonian.

        """
        Hs = 0 * I ^ self.nqubits
        for i in range(self.nqubits):
            for j in self.graph[i]:
                Hs += (
                    (I ^ self.nqubits) - ((I ^ self.nqubits - i - 1) ^ Z ^ (I ^ i))
                ) @ ((I ^ self.nqubits) + ((I ^ self.nqubits - j - 1) ^ Z ^ (I ^ j)))
        return 0.25 * Hs

    def gen_Hp(self) -> PauliOp:
        """Generate the profit Hamiltonian
        :math:`H_{p}=\sum_{i}w(i)(1-Z_{i})/2`

        Returns
        ----------
        qiskit.opflow.PauliOp
            Profit Hamiltonian.

        """
        Hp = 0 * I ^ self.nqubits
        for i in range(self.nqubits):
            # Qiskit opflow doesn't support multiplying types other than python built-in ones (e.g. np.float64)
            Hp += float(self.dat[self.idx2cord[i]]) * (
                (I ^ self.nqubits) - ((I ^ self.nqubits - i - 1) ^ Z ^ (I ^ i))
            )
        return 0.5 * Hp

    def gen_Hamiltonian(self, penalty: Union[float, bool, None]) -> PauliOp:
        """Generate the Hamiltonian with penalty weight :math:`\gamma`.

        :math:`H=-H_{p}+\gamma H_{s}`

        Parameter   s
        ----------
        penalty : float
            Penalty for the smoothness term in the Hamiltonian.

        Returns
        ----------
        qiskit.opflow.PauliOp
            Hamiltonian with penalty weight :math:`\gamma`.

        """
        if penalty is True:
            penalty = self.heuristic_penalty()
        if penalty:
            return (-self.Hp + penalty * self.Hs).reduce()
        return self.gen_projected_Hamiltonian()

    def gen_projected_Hamiltonian(self) -> PauliOp:
        """Generate the profit Hamiltonian projected onto valid states.

        Returns
        ----------
        qiskit.opflow.PauliOp
            :math:`H=-PH_{p}P`

        """
        state_fn = (-self.Hs @ (Plus ^ self.nqubits)).reduce().eval().to_dict_fn()
        self.valid_configs = [
            int(k, 2) for k, v in state_fn.primitive.items() if abs(v) < 1e-8
        ]
        p_op = np.zeros((2 ** self.nqubits))
        p_op[np.array(self.valid_configs, dtype=int)] = 1
        p_op = MatrixOp(np.diag(p_op))

        return (p_op @ -self.Hp @ p_op).reduce().to_matrix_op()

    def gen_pseudoflow_graph(self, MAX_FLOW: int) -> Tuple[nx.DiGraph, int, int]:
        G = nx.DiGraph()
        G.add_nodes_from(np.arange(self.nqubits))
        source = -1
        sink = self.nqubits

        for p in self.graph:
            for c in self.graph[p]:
                G.add_edge(p, c, const=MAX_FLOW)

            if self.dat[self.idx2cord[p]] >= 0:
                G.add_edge(source, p, const=self.dat[self.idx2cord[p]])
            else:
                G.add_edge(p, sink, const=-self.dat[self.idx2cord[p]])

        return G, source, sink

    def plot_mine_state(
        self, bitstring: str, bit_ordering: Optional[str] = "R"
    ) -> None:
        """Plot the mining state represented by the bitstring.

        Parameters
        ----------
        bitstring : str
            A 0/1 string represents the state. Length of the string is the same as the number of qubits.
        bit_ordering : str
            Available options ``[\'L\', \'R\']``. ``\'L\'`` means the least significant bit (LSB) is on the left, and ``\'R\'`` means LSB is on the right. LSB represents the qubit with index 0.
        """
        assert (
            len(bitstring) == self.nqubits
        ), "Length of the bitstring should be the same as the number of qubits."
        assert bit_ordering in ["L", "R"], "bit_ordering options: 'L', 'R'."
        if bit_ordering == "R":
            bitstring = "".join(list(bitstring)[::-1])

        x = PrettyTable([" "] + [str(ic) for ic in range(self.cols)])
        for ir in range(self.rows):
            x.add_row(
                [ir]
                + [
                    bitstring[self.cord2idx[(ir, ic)]]
                    if (ir, ic) in self.cord2idx
                    else "x"
                    for ic in range(self.cols)
                ]
            )
        print(str(x))

    def get_profit(self, bitstring: str, bit_ordering: Optional[str] = "R") -> int:
        """Return profit for a vector state.

        Parameters
        ----------
        bitstring : str
            A 0/1 string represents the state. Length of the string is the same as the number of qubits.
        bit_ordering : str
            Available options ``[\'L\', \'R\']``. ``\'L\'`` means the least significant bit (LSB) is on the left, and ``\'R\'`` means LSB is on the right. LSB represents the qubit with index 0.
        """
        assert (
            len(bitstring) == self.nqubits
        ), "Length of the bitstring should be the same as the number of qubits."
        assert bit_ordering in ["L", "R"], "bit_ordering options: 'L', 'R'."
        if bit_ordering == "R":
            bitstring = "".join(list(bitstring)[::-1])

        return sum(
            [
                self.dat[self.idx2cord[i]]
                for i in range(self.nqubits)
                if bitstring[i] == "1"
            ]
        )

    def get_violation(self, bitstring: str, bit_ordering: Optional[str] = "R") -> int:
        """Return violation for a vector state.

        Parameters
        ----------
        bitstring : str
            A 0/1 string represents the state. Length of the string is the same as the number of qubits.
        bit_ordering : str
            Available options ``[\'L\', \'R\']``. ``\'L\'`` means the least significant bit (LSB) is on the left, and ``\'R\'`` means LSB is on the right. LSB represents the qubit with index 0.
        """
        assert (
            len(bitstring) == self.nqubits
        ), "Length of the bitstring should be the same as the number of qubits."
        assert bit_ordering in ["L", "R"], "bit_ordering options: 'L', 'R'."
        if bit_ordering == "R":
            bitstring = "".join(list(bitstring)[::-1])

        dig = list(map(lambda x: -1 if x == "1" else 1, list(bitstring)))
        res = 0
        for i in range(self.nqubits):
            for j in self.graph[i]:
                res += 0.25 * (1.0 - dig[i]) * (1.0 + dig[j])
        return int(res)

    def heuristic_penalty(self, coeff: float = 3.8) -> float:
        return (
            float(
                np.linalg.norm(np.where(self.dat.flat != np.inf), ord=2) / self.nqubits
            )
            * coeff
        )

    def solve(
        self,
        algorithm: Union[MinimumEigensolver, Pseudoflow],
        penalty: Union[float, bool, None] = None,
        benchmark: Union[Benchmark, bool, None] = None,
    ) -> "MiningProblemResult":
        if benchmark is True:
            benchmark = Benchmark()
        elif benchmark is False or benchmark is None:
            benchmark = Benchmark(activate=False)

        with benchmark:
            self._ret = MiningProblemResult()

            if isinstance(algorithm, MinimumEigensolver):
                res = algorithm.compute_minimum_eigenvalue(
                    self.gen_Hamiltonian(penalty), [self.Hp, self.Hs]
                )

                if isinstance(algorithm, NumPyMinimumEigensolver):
                    self._ret.optimal_config = format(
                        np.argwhere(res.eigenstate.primitive.data.real == 1.0).item(),
                        f"0{self.nqubits + 2}b",
                    )[2:]
                    self._ret.optimal_config_prob = 1.0

                else:
                    if isinstance(res.eigenstate, dict):
                        self._ret.optimal_config, self._ret.optimal_config_prob = max(
                            (
                                item
                                for item in res.eigenstate.items()
                                if self.valid_configs is None
                                or int(item[0], 2) in self.valid_configs
                            ),
                            key=lambda item: item[1],
                        )
                    elif isinstance(res.eigenstate, np.ndarray):
                        idx = np.argmax(res.eigenstate * res.eigenstate.conj())
                        self._ret.optimal_config = int_to_bitstr(idx, self.nqubits)
                        self._ret.optimal_config_prob = res.eigenstate[idx]

                    if isinstance(self._ret.optimal_config_prob, complex):
                        self._ret.optimal_config_prob = (
                            self._ret.optimal_config_prob.real ** 2
                            - self._ret.optimal_config_prob.imag ** 2
                        )
                    else:
                        self._ret.optimal_config_prob **= 2

                self._ret.ground_state = res.eigenstate
                if (
                    res.aux_operator_eigenvalues is not None
                ):  # Remove after implemented ASP expectation value
                    (
                        self._ret._expected_profit,
                        self._ret._expected_violation,
                    ) = res.aux_operator_eigenvalues

            elif isinstance(algorithm, Pseudoflow):
                self._ret.optimal_config = algorithm.run(
                    *self.gen_pseudoflow_graph(algorithm.MAX_FLOW)
                )
                self._ret.optimal_config_prob = 1.0

            elif isinstance(algorithm, PEPSITE):
                self._ret.optimal_config = algorithm.run(self)
                self._ret.optimal_config_prob = 1.0
                
            else:
                raise ValueError(f"{type(algorithm)} is not a valid algorithm.")

            print(
                "The most probable configuration and the corresponding probability:"
                f" {self._ret.optimal_config, self._ret.optimal_config_prob}"
            )
            self.plot_mine_state(self._ret.optimal_config)
            return self._ret


class SubMine(BaseMine):
    def __init__(self, mine: Mine, node_list: List) -> None:
        self.node_list = node_list
        self.nqubits = len(self.node_list)
        self.node_c = {}  # key: a node in this frag, value: child node in another frag
        self.node_p = {}  # key: a node in this frag, value: parent node in another frag
        self.node_bd = []  # submine idx of bd nodes
        self.zb = []  # pauli z op of bd nodes

        self._init_nodes(mine)
        self.Hp = self.gen_Hp(mine)  # profit
        self.Hs = self.gen_Hs(mine)  # smoothness, inner nodes
        self.Hb = null_operator(self.nqubits)  # boundary

    def _init_nodes(self, mine: Mine) -> None:
        for idx, i in enumerate(self.node_list):
            self.node_c[i] = [j for j in mine.graph[i] if j not in self.node_list]
            self.node_p[i] = [j for j in mine.graph_r[i] if j not in self.node_list]
            if len(self.node_c[i]) + len(self.node_p[i]) > 0:
                self.node_bd.append(idx)
                self.zb.append(single_qubit_pauli("z", idx, self.nqubits))

    def gen_Hs(self, mine: Mine) -> PauliOp:
        Hs = null_operator(self.nqubits)
        for sub_i, i in enumerate(self.node_list):
            for j in mine.graph[i]:
                try:
                    sub_j = self.node_list.index(j)
                    Hs += z_projector(1, sub_i, self.nqubits) @ z_projector(
                        0, sub_j, self.nqubits
                    )
                except:
                    continue
        return Hs

    def gen_Hp(self, mine: Mine) -> PauliOp:
        Hp = null_operator(self.nqubits)
        for sub_idx, idx in enumerate(self.node_list):
            Hp += float(mine.dat[mine.idx2cord[idx]]) * z_projector(
                1, sub_idx, self.nqubits
            )
        return Hp

    def _update_Hb(self, z_val: Dict) -> None:
        self.Hb = null_operator(self.nqubits)
        for sub_idx in self.node_bd:
            mine_idx = self.node_list[sub_idx]
            for j in self.node_c[mine_idx]:
                self.Hb += float(0.5 * (1 + z_val[j])) * z_projector(
                    1, sub_idx, self.nqubits
                )
            for j in self.node_p[mine_idx]:
                self.Hb += float(0.5 * (1 - z_val[j])) * z_projector(
                    0, sub_idx, self.nqubits
                )

    def gen_Hamiltonian(self, penalty: float, z_val: Dict) -> PauliOp:
        self._update_Hb(z_val)
        H_res = -self.Hp + penalty * (self.Hs + self.Hb)
        return H_res

    def solve(
        self, algorithm: MinimumEigensolver, penalty: Union[float, None], z_val: Dict
    ) -> "MiningProblemResult":
        if isinstance(algorithm, MinimumEigensolver):
            res = algorithm.compute_minimum_eigenvalue(
                self.gen_Hamiltonian(penalty, z_val), self.zb + [self.Hp, self.Hs]
            )
            self._ret = MiningProblemResult()
            self._ret.optimal_config, self._ret.optimal_config_prob = max(
                res.eigenstate.items(), key=lambda item: item[1]
            )
            # self._ret.ground_state = res.eigenstate
            for sub_idx, zi in zip(self.node_bd, res.aux_operator_eigenvalues[:-2]):
                z_val[self.node_list[sub_idx]] = zi
            self._ret.expected_profit = res.aux_operator_eigenvalues[-2]
            self._ret.expected_violation = res.aux_operator_eigenvalues[-1]
            return self._ret
        else:
            raise ValueError()


class SubMine(BaseMine):
    def __init__(self, mine: Mine, node_list: List) -> None:
        super.__init__()
        self.node_list = node_list
        self.nqubits = len(self.node_list)
        self.node_c = {}  # key: a node in this frag, value: child node in another frag
        self.node_p = {}  # key: a node in this frag, value: parent node in another frag
        self.node_bd = []  # submine idx of bd nodes
        self.zb = []  # pauli z op of bd nodes

        self._init_nodes(mine)
        self.Hp = self.gen_Hp(mine)  # profit
        self.Hs = self.gen_Hs(mine)  # smoothness, inner nodes
        self.Hb = null_operator(self.nqubits)  # boundary

    def _init_nodes(self, mine: Mine) -> None:
        for idx, i in enumerate(self.node_list):
            self.node_c[i] = [j for j in mine.graph[i] if j not in self.node_list]
            self.node_p[i] = [j for j in mine.graph_r[i] if j not in self.node_list]
            if len(self.node_c[i]) + len(self.node_p[i]) > 0:
                self.node_bd.append(idx)
                self.zb.append(single_qubit_pauli("z", idx, self.nqubits))

    def gen_Hs(self, mine: Mine) -> PauliOp:
        Hs = null_operator(self.nqubits)
        for sub_i, i in enumerate(self.node_list):
            for j in mine.graph[i]:
                try:
                    sub_j = self.node_list.index(j)
                    Hs += z_projector(1, sub_i, self.nqubits) @ z_projector(
                        0, sub_j, self.nqubits
                    )
                except:
                    continue
        return Hs

    def gen_Hp(self, mine: Mine) -> PauliOp:
        Hp = null_operator(self.nqubits)
        for sub_idx, idx in enumerate(self.node_list):
            Hp += float(mine.dat[mine.idx2cord[idx]]) * z_projector(
                1, sub_idx, self.nqubits
            )
        return Hp

    def _update_Hb(self, z_val: Dict) -> None:
        self.Hb = null_operator(self.nqubits)
        for sub_idx in self.node_bd:
            mine_idx = self.node_list[sub_idx]
            for j in self.node_c[mine_idx]:
                self.Hb += float(0.5 * (1 + z_val[j])) * z_projector(
                    1, sub_idx, self.nqubits
                )
            for j in self.node_p[mine_idx]:
                self.Hb += float(0.5 * (1 - z_val[j])) * z_projector(
                    0, sub_idx, self.nqubits
                )

    def gen_Hamiltonian(self, penalty: float, z_val: Dict) -> PauliOp:
        self._update_Hb(z_val)
        H_res = -self.Hp + penalty * (self.Hs + self.Hb)
        return H_res

    def solve(
        self, algorithm: MinimumEigensolver, penalty: Union[float, None], z_val: Dict
    ) -> "MiningProblemResult":
        if isinstance(algorithm, MinimumEigensolver):
            res = algorithm.compute_minimum_eigenvalue(
                self.gen_Hamiltonian(penalty, z_val), self.zb + [self.Hp, self.Hs]
            )
            self._ret = MiningProblemResult()
            self._ret.optimal_config, self._ret.optimal_config_prob = max(
                res.eigenstate.items(), key=lambda item: item[1]
            )
            # self._ret.ground_state = res.eigenstate
            for sub_idx, zi in zip(self.node_bd, res.aux_operator_eigenvalues[:-2]):
                z_val[self.node_list[sub_idx]] = zi
            self._ret.expected_profit = res.aux_operator_eigenvalues[-2]
            self._ret.expected_violation = res.aux_operator_eigenvalues[-1]
            return self._ret
        else:
            raise ValueError()


class MiningProblemResult(AlgorithmResult):
    def __init__(self) -> None:
        super().__init__()
        self._optimal_config = None
        self._optimal_config_prob = None
        self._ground_state = None
        self._expected_profit = None
        self._expected_violation = None

    @property
    def optimal_config(self) -> Optional[str]:
        return self._optimal_config

    @optimal_config.setter
    def optimal_config(self, optimal_config: str) -> None:
        self._optimal_config = optimal_config

    @property
    def optimal_config_prob(self) -> Optional[float]:
        return self._optimal_config_prob

    @optimal_config_prob.setter
    def optimal_config_prob(self, optimal_config_prob: float) -> None:
        self._optimal_config_prob = optimal_config_prob

    @property
    def ground_state(self) -> Optional[np.ndarray]:
        return self._ground_state

    @ground_state.setter
    def ground_state(self, ground_state: np.ndarray) -> None:
        self._ground_state = ground_state

    @property
    def expected_profit(self) -> Optional[float]:
        return self._expected_profit

    @expected_profit.setter
    def expected_profit(self, expected_profit: float) -> None:
        self._expected_profit = expected_profit

    @property
    def expected_violation(self) -> Optional[float]:
        return self._expected_violation

    @expected_violation.setter
    def expected_violation(self, expected_violation: float) -> None:
        self._expected_violation = expected_violation
