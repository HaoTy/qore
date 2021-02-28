"""The Adiabatic State Preparation algorithm.

See https://arxiv.org/pdf/quant-ph/0001106.pdf
"""

from typing import Optional, Union, Dict
import numpy as np
from qiskit.aqua.algorithms import AlgorithmResult, QuantumAlgorithm


class ASP(QuantumAlgorithm):
    """
    The Adiabatic State Preparation algorithm.

    See https://arxiv.org/pdf/quant-ph/0001106.pdf
    """
    def __init__(self) -> None:
        raise NotImplementedError()

    def _run(self) -> Dict:
        raise NotImplementedError()


