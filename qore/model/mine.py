"""The Mine class."""

__docformat__ = "reStructuredText"

from typing import Optional, Union, Callable
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from prettytable import PrettyTable
from qiskit.opflow import PauliOp, I, Z, Plus, MatrixOp
from qiskit.algorithms import MinimumEigensolver, AlgorithmResult

from ..algorithms import Pseudoflow


class Mine:
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
        self.graph = defaultdict(list)
        self.idx2cord = []
        self.cord2idx = {}
        self._init_mapping()
        self.nqubits: int = len(self.idx2cord)
        self.valid_configs = None

        self.Hs = self.gen_Hs()
        self.Hp = self.gen_Hp()

    def _init_mapping(self) -> None:
        """Assign a unique id to each valid node and store the graph structure with ``self.graph``."""
        for r in range(self.rows):
            for c in range(self.cols):
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

    def gen_Hamiltonian(self, penalty: Union[float, None]) -> PauliOp:
        """Generate the Hamiltonian with penalty weight :math:`\gamma`.

        :math:`H=-H_{p}+\gamma H_{s}`

        Parameters
        ----------
        penalty : float
            Penalty for the smoothness term in the Hamiltonian.

        Returns
        ----------
        qiskit.opflow.PauliOp
            Hamiltonian with penalty weight :math:`\gamma`.

        """
        if penalty:
            return (-self.Hp + penalty * self.Hs).reduce()
        return self.gen_projected_Hamiltonian()

    def gen_projected_Hamiltonian(self) -> PauliOp:
        self.valid_configs = set(range(2 ** self.nqubits)) - set(
            int(k, 2)
            for k in (-self.Hs @ (Plus ^ self.nqubits))
            # .to_matrix_op()
            .reduce()
            .eval()
            .sample(shots=2 ** (self.nqubits + 6))
            .keys()
        )
        p_op = np.zeros((2 ** self.nqubits))
        p_op[np.array(list(self.valid_configs), dtype=int)] = 1
        p_op = MatrixOp(np.diag(p_op))
        return (p_op @ -self.Hp @ p_op).reduce().to_matrix_op()

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

    def solve(
        self,
        algorithm: Union[MinimumEigensolver, Pseudoflow],
        penalty: Union[float, None] = None,
    ) -> "MiningProblemResult":
        if isinstance(algorithm, MinimumEigensolver):
            res = algorithm.compute_minimum_eigenvalue(
                self.gen_Hamiltonian(penalty), [self.Hp, self.Hs]
            )
            self._ret = MiningProblemResult()
            self._ret.optimal_config, self._ret.optimal_config_prob = max(
                (
                    item
                    for item in res.eigenstate.items()
                    if int(item[0], 2) in self.valid_configs
                ),
                key=lambda item: item[1],
            )
            self._ret.ground_state = res.eigenstate
            if (
                res.aux_operator_eigenvalues is not None
            ):  # Remove after implemented ASP expectation value
                (
                    self._ret._expected_profit,
                    self._ret._expected_violation,
                ) = res.aux_operator_eigenvalues
            return self._ret
        elif isinstance(algorithm, Pseudoflow):
            raise NotImplementedError()
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
