""" The pseudoflow algorithm for open-pit mining problems.

See https://hochbaum.ieor.berkeley.edu/html/pub/Hochbaum-OR.pdf
"""

from networkx import DiGraph
from numpy import MAXDIMS
from pseudoflow import hpf


class Pseudoflow:
    def __init__(self, MAX_FLOW: int = 1000000) -> None:
        self._MAX_FLOW = MAX_FLOW

    @property
    def MAX_FLOW(self):
        return self._MAX_FLOW

    def run(self, graph, source, sink, verbose=False) -> str:
        _, cuts, info = hpf(graph, source, sink, const_cap="const")
        if verbose:
            print(info)
        ground_state = [value[0] for _, value in cuts.items()][1:-1]
        bitstring = "".join(list(map(str, ground_state[::-1])))
        return bitstring
