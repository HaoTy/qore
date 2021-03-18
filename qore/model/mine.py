"""The Mine class."""

__docformat__ = 'reStructuredText'

from typing import List, Union
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable
from qiskit.aqua.operators import PauliOp, WeightedPauliOperator

from qore.utils import null_operator, z_projector


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
                self.dat = np.loadtxt(mine_config)
            except:
                raise IOError('Invalid Mine Configuration File')
        elif isinstance(mine_config, np.ndarray):
            if len(mine_config.shape) != 2:
                raise ValueError('`mine_config` must be two-demensional')
            self.dat = mine_config
        else:
            raise ValueError('Unrecognized `mine_config` type')

        self.rows, self.cols = self.dat.shape
        self.graph = defaultdict(list)
        self.idx2cord = []
        self.cord2idx = {}
        self._init_mapping()
        self.nqubits: int = len(self.idx2cord)

        self.Hs = self.gen_Hs()
        self.Hp = self.gen_Hp()

    def _init_mapping(self) -> None:
        """Assign a unique id to each valid node and store the graph structure with ``self.graph``.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if self.dat[r, c] < float('inf'):
                    self.idx2cord.append((r, c))
                    self.cord2idx[(r, c)] = len(self.idx2cord)-1
                    idx = self.cord2idx[(r, c)]
                    for pr, pc in [(r-1, c-1), (r-1, c), (r-1, c+1)]:
                        if 0 <= pr < self.rows and 0 <= pc < self.cols and self.dat[pr, pc] < float('inf'):
                            self.graph[idx].append(self.cord2idx[(pr, pc)])

    def plot_mine(self) -> None:
        """Plot the mine configuration.
        """
        x = PrettyTable([' ']+[str(ic) for ic in range(self.cols)])
        for ir in range(self.rows):
            x.add_row([ir]+['%.3f' % self.dat[ir, ic]
                            for ic in range(self.cols)])
        print(str(x))

    def gen_Hs(self) -> Union[PauliOp, WeightedPauliOperator]:
        """Generate the smoothess Hamiltonian 
        :math:`H_{s}=\sum_{i}\sum_{j:Parent(i)} (1-Z_{i})/2*(1+Z_{j})/2`

        Returns
        ----------
        Union[PauliOp, WeightedPauliOperator]
            Smoothness Hamiltonian.

        """
        Hs = null_operator(self.nqubits)
        for i in range(self.nqubits):
            for j in self.graph[i]:
                Hs += z_projector(1, i, self.nqubits) * \
                    z_projector(0, j, self.nqubits)
        return Hs

    def gen_Hp(self) -> Union[PauliOp, WeightedPauliOperator]:
        """Generate the profit Hamiltonian 
        :math:`H_{p}=\sum_{i}w(i)(1-Z_{i})/2`

        Returns
        ----------
        Union[PauliOp, WeightedPauliOperator]
            Profit Hamiltonian.

        """
        Hp = null_operator(self.nqubits)
        for i in range(self.nqubits):
            Hp += self.dat[self.idx2cord[i]]*z_projector(1, i, self.nqubits)
        return Hp

    def gen_Hamiltonian(self, penalty: float) -> Union[PauliOp, WeightedPauliOperator]:
        """Generate the Hamiltonian with penalty weight :math:`\gamma`.

        :math:`H=H_{s}+\gamma H_{p}`

        Parameters
        ----------
        penalty : float
            Penalty for the smoothness term in the Hamiltonian. 

        Returns
        ----------
        Union[PauliOp, WeightedPauliOperator]
            Hamiltonian with penalty weight :math:`\gamma`.

        """
        return -self.Hp + penalty*self.Hs

    def plot_mine_state(self, bitstring) -> None:
        """Plot the mining state represented by the bitstring.

        Parameters
        ----------
        bitstring : str
            A 0/1 string represents the state. Length of the string is the same as the number of qubits.
        """
        assert len(
            bitstring) == self.nqubits, "Length of the bitstring should be the same as the number of qubits."

        x = PrettyTable([' ']+[str(ic) for ic in range(self.cols)])
        for ir in range(self.rows):
            x.add_row([ir]+[bitstring[self.cord2idx[(ir, ic)]] if (ir, ic) in self.cord2idx else 'x'
                            for ic in range(self.cols)])
        print(str(x))

    def get_profit(self, bitstring: str) -> int:
        """Return profit for a vector state.

        Parameters
        ----------
        bitstring : str
            A 0/1 string represents the state. Length of the string is the same as the number of qubits.
        """
        assert len(
            bitstring) == self.nqubits, "Length of the bitstring should be the same as the number of qubits."
        return sum([self.dat[self.idx2cord[i]] for i in range(self.nqubits) if bitstring[i] == '1'])

    def get_violation(self, bitstring: str) -> int:
        """Return violation for a vector state.

        Parameters
        ----------
        bitstring : str
            A 0/1 string represents the state. Length of the string is the same as the number of qubits.
        """
        assert len(
            bitstring) == self.nqubits, "Length of the bitstring should be the same as the number of qubits."
        dig = list(map(lambda x: 1 if x=='1' else -1, list(bitstring)))
        res = 0
        for i in range(self.nqubits):
            for j in self.graph[i]:
                res += 0.25* (1. + dig[i]) * (1. - dig[j])
        return res
