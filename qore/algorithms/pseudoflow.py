""" The pseudoflow algorithm for open-pit mining problems.

See https://hochbaum.ieor.berkeley.edu/html/pub/Hochbaum-OR.pdf
"""

from typing import Optional, Union, Dict, List, Callable
import networkx as nx
from networkx import Graph
from networkx.classes.digraph import DiGraph
import pseudoflow
# import numpy as np

class Pseudoflow():
    def __init__(self,
                 graph: DiGraph,
                 source: int,
                 sink: int,
                 ) -> None:
        self.graph = graph
        self.source = source
        self.sink = sink

    def run(self, verbose=False) -> str:
        breakpoints, cuts, info = pseudoflow.hpf(
            self.graph,
            self.source,
            self.sink,
            const_cap='const')
        if verbose: print(info)
        ground_state = [value[0] for _, value in cuts.items()][1:-1]
        bitstring = "".join(list(map(str, ground_state[::-1])))
        return {'ground_state': bitstring}
        

