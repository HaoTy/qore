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

    def __init__(self, mine_config: str = '') -> None:
        """Initialize the Mine class.
        :param mine_config: path to a .txt file which stores the mine configuration 
        :type mine_config: str
        """
        try:
            self.dat = np.loadtxt(mine_config)
        except:
            raise Exception('Invalid Mine Configuration File')

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
        """Plot the mine.
        """
        x = PrettyTable([' ']+[str(ic) for ic in range(self.cols)])
        for ir in range(self.rows):
            x.add_row([ir]+['%.3f' % self.dat[ir, ic]
                            for ic in range(self.cols)])
        print(str(x))

    def gen_Hs(self) -> Union[PauliOp, WeightedPauliOperator]:
        r"""Generate the smoothess Hamiltonian 
        :math:`H_{s}=\sum_{i}\sum_{j:Parent(i)} (1-Z_{i})/2*(1+Z_{j})/2`
        """
        Hs = null_operator(self.nqubits)
        for i in range(self.nqubits):
            for j in self.graph[i]:
                Hs += z_projector(1, i, self.nqubits) * \
                    z_projector(0, j, self.nqubits)
        return Hs

    def gen_Hp(self) -> Union[PauliOp, WeightedPauliOperator]:
        r"""Generate the profit Hamiltonian 
        :math:`H_{p}=\sum_{i}w(i)(1-Z_{i})/2`
        """
        Hp = null_operator(self.nqubits)
        for i in range(self.nqubits):
            Hp += self.dat[self.idx2cord[i]]*z_projector(1, i, self.nqubits)
        return Hp

    def gen_Hamiltonian(self, penalty: float) -> Union[PauliOp, WeightedPauliOperator]:
        r"""Generate the Hamiltonian with penalty weight :math:`\gamma`.

        :math:`H=H_{s}+\gamma H_{p}`

        :param penalty: penalty for the smoothness term in the Hamiltonian 
        :type penalty: float
        """
        return self.Hp + penalty*self.Hs
