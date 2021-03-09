"""The Mine class."""

__docformat__ = 'reStructuredText'

from typing import List, Optional, Union, Tuple
from collections import defaultdict

import numpy as np
from prettytable import PrettyTable

from .hamiltonian import null_operator, z_projector


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

        self.Hs = self._gen_Hs()
        self.Hp = self._gen_Hp()

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

    def _gen_Hs(self):
        r"""Generate the smoothess Hamiltonian ``Hs``.
        :math:`H_{s}=\sum_{i}\sum_{j:Parent of i} (1-Z_{i})/2*(1+Z_{j})/2`,
        """
        Hs = null_operator(self.nqubits)
        for i in range(self.nqubits):
            for j in self.graph[i]:
                Hs += z_projector(1, i, self.nqubits) * \
                    z_projector(0, j, self.nqubits)
        return Hs

    def _gen_Hp(self):
        r"""Generate the profit Hamiltonian ``Hp``.
        :math:`H_{p}=\sum_{i}w(i)(1-Z_{i})/2`
        """
        Hp = null_operator(self.nqubits)
        for i in range(self.nqubits):
            Hp += self.dat[self.idx2cord[i]]*z_projector(1, i, self.nqubits)
        return Hp

    def gen_Hamiltonian(self, penalty: float):
        """Generate the Hamiltonian H=Hs+penalty*Hp.
        :param penalty: penalty for the smoothness term in the Hamiltonian 
        :type penalty: float
        """
        return self.Hp + penalty*self.Hs
