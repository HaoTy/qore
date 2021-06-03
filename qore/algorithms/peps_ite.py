"""
Projected Entangled Pair States (PEPS) with Imaginary Time Evolution (ITE).
"""
from koala import peps, Observable
import numpy as np
import tensorbackends as tb

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import Mine



class PEPSITE:
    """
    Projected Entangled Pair States (PEPS) with Imaginary Time Evolution (ITE).
    """

    def __init__(
        self,
        evol_time: float,
        contract_option: Optional[peps.contraction.ContractOption] = None,
        backend: Optional[Union[str, tb.backends.Backend]] = None,
    ) -> None:
        """
        Initialize the PEPS ITE algorithm.

        Parameters
        ----------
        """
        self._evol_time = evol_time
        self.backend = backend if backend else "numpy"
        self._contract_option = contract_option if contract_option else peps.Snake()
        self._peps = None
        self._mine = None
        self._ret = None

    @property
    def evol_time(self) -> float:
        return self._evol_time

    @evol_time.setter
    def evol_time(self, evol_time: float) -> None:
        self._evol_time = evol_time

    @property
    def backend(self) -> tb.backends.Backend:
        """ Returns backend. """
        return self._backend

    @backend.setter
    def backend(self, backend: Union[str, tb.backends.Backend]) -> None:
        """ Sets backend. """
        self._backend = tb.get(backend)

    @property
    def contract_option(self) -> Optional[peps.contraction.ContractOption]:
        """ Returns contract option. """
        return self._contract_option

    @contract_option.setter
    def contract_option(self, contract_option: peps.contraction.ContractOption) -> None:
        """ Sets contract option."""
        self._contract_option = contract_option

    def _construct_peps(self) -> peps.PEPS:
        backend = self._backend
        n = self._mine.dat.shape[1]
        p = np.empty((n, n), dtype=object)

        # top-left corner
        p[0, 0] = backend.astensor(np.eye(2, dtype=complex).reshape(1, 1, 2, 1, 2, 1))

        # bottom-right corner
        p[-1, -1] = backend.astensor(np.eye(2, dtype=complex).reshape(1, 1, 1, 2, 2, 1))

        # diagonal
        tsr = np.zeros((2, 2, 2), dtype=complex)
        tsr[0, 0, 0] = 1
        tsr[1, 1, 1] = 1
        tsr = tsr.reshape(1, 1, 2, 2, 2, 1)
        for i in range(1, n - 1):
            p[i, i] = backend.astensor(tsr.copy())

        # bottom-left corner
        p[-1, 0] = backend.astensor(tsr.reshape(2, 2, 1, 1, 2, 1))

        # left edge
        tsr = np.zeros((2, 2, 2), dtype=complex)
        tsr[:, :, 0] = np.ones((2, 2), dtype=complex)
        tsr[1, 1, 1] = 1
        tsr = tsr.reshape(2, 2, 2, 1, 1, 1)
        for i in range(1, n - 1, 2):
            p[i, 0] = backend.astensor(tsr.copy())

        # bottom edge
        tsr = tsr.reshape(2, 2, 1, 2, 1, 1)
        for j in range(1, n - 1, 2):
            p[-1, j] = backend.astensor(tsr.copy())

        # left edge
        tsr = np.zeros((2, 2, 2, 2), dtype=complex)
        tsr[0, 0, 0, 0] = 1
        tsr[1, 1, 1, 1] = 1
        tsr = tsr.reshape(2, 2, 2, 1, 2, 1)
        for i in range(2, n - 1, 2):
            p[i, 0] = backend.astensor(tsr.copy())

        # bottom edge
        tsr = tsr.reshape(2, 2, 1, 2, 2, 1)
        for j in range(2, n - 1, 2):
            p[-1, j] = backend.astensor(tsr.copy())

        # middle tensors
        tsr = [
            np.zeros((2, 2, 2, 2, 2), dtype=complex),
            np.zeros((2, 2, 2, 2), dtype=complex),
        ]
        tsr[0][0, 0, 0, 0, 0] = 1
        tsr[0][1, 1, 1, 1, 1] = 1
        tsr[0] = tsr[0].reshape(2, 2, 2, 2, 2, 1)
        tsr[1][:, :, 0, 0] = np.ones((2, 2), dtype=complex)
        tsr[1][1, 1, :, :] = np.ones((2, 2), dtype=complex)
        tsr[1] = tsr[1].reshape(2, 2, 2, 2, 1, 1)
        for i in range(2, n - 1):
            for j in range(1, i):
                p[i, j] = backend.astensor(tsr[abs(i - j) % 2].copy())

        # fill in the rest
        tsr = np.ones((1, 1, 1, 1, 1, 1), dtype=complex)
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                p[i, j] = backend.astensor(tsr.copy())

        self._peps = peps.PEPS(p, backend)
        return self._peps

    # def _transform_coordinate(self, i: int, j: int) -> int:
    # return (i + j) * self._peps.ncol + j - i

    def _ite(self) -> float:
        for q in range(self._mine.nqubits):
            i, j = self._mine.idx2cord[q]
            self._peps.apply_operator(
                self._backend.astensor(
                    np.array(
                        [1, 0, 0, np.exp(self._evol_time * self._mine.dat[i, j])],
                        dtype=complex,
                    ).reshape(2, 2)
                ),
                [(i + j) * self._peps.ncol + j - i],
            )
        return self._peps

    def run(self, mine: "Mine") -> str:
        self._mine = mine
        self._construct_peps()
        self._ite()

        bitstr = ""
        for q in range(mine.nqubits):
            i, j = mine.idx2cord[q]
            expectation = self._peps.expectation(
                Observable.Z((i + j) * self._peps.ncol + j - i),
                use_cache=isinstance(self._contract_option, peps.BMPS),
                contract_option=self._contract_option,
            )
            bitstr += "1" if expectation < 0 else "0"
        self._ret = bitstr[::-1]
        return self._ret
