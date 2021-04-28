# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from qiskit.providers.aer import QasmSimulator

from qore import Mine, ASP, QAOA, VQE, Pseudoflow, ExactDiagonalization

# %% [markdown]
# ### First, we define the problem with a 2D numpy array or a file path.
# "inf" represents undiggable sites.
# 
# Commented lines are some simpler mine problems you can try, as the current one is non-trivial to solve and some algorithms may fail under certain circumstances.
# 
# In addition, you may define your own mine problems to play with.

# %%
# qmine = Mine('mine_config.txt')

# qmine = Mine(np.array([[-2., 3., 1., -1.], [float('inf'), 5., 3., float('inf')]]))

qmine = Mine(np.array([[-2.0, 3.0, -1.0, -2.0, -1.0], [float('inf'), 1.0, -5.0, 10.0, float('inf')], [float('inf'), float('inf'), 4.0, float('inf'), float('inf')]]))

qmine.plot_mine()

# %% [markdown]
# ### Problem Abstraction 
# 
# The mining problem can be treated as a classical max flow problem in graph theory. Alternatively, it can be mapped to a quantum problem via constructing an appropriate Hamiltonian. A typical choice is to use Ising spin states to represent dug/undug states for each node. $|0\rangle$ means undug and $|1\rangle$ means dug. Then the profit term $H_{p}$ should be a sum of weighted Pauli matrices in Z direction ($Z$), and the mining constraints are encoded in the smoothness term $H_{s}$. The tunning parameter $\lambda$ controls the "importance" of the smoothness term: a high $\lambda$ could result in trivial mining configurations, while a low $\lambda$ could result in invalid configurations with high profit. In practice, $\lambda$ is decided empirically for each mining problem. We may refer $\lambda$ as the "penalty" factor in the following instructions.
# 
# $H=-H_{p}+\lambda H_{s}$
# 
# $H_{p}=\sum_{i}w_{i} (I-Z_{i})/2$
# 
# $H_{s}=\sum_{i,j=p(i)}(I+Z_{j})/2*(I-Z_{i})/2$

# %%
print(qmine.Hs)
print(qmine.Hp)

# %% [markdown]
# ### Following we define the algorithm to use with proper parameters.
# 
# Available algorithms: 
# - Classical: Pseudoflow, ExactDiagonalization
# - Quantum: ASP, QAOA, VQE
# 
# Pseudoflow is a state-of-the-art classical algorithm for the open-mining problem. It is fast and can be used to check the correctness of the results given by quantum algorithms.
# 
# ExactDiagonalization is directly diagonalizing the Hamiltonian to find its eigenvectors. It is fast for small systems but intractable for large systems.
# 
# ASP (Adiabatic State Preparation, https://arxiv.org/abs/quant-ph/0001106), QAOA (Quantum Approximate Optimization Algorithm, https://arxiv.org/abs/1411.4028), and VQE (Variational Quantum Eigensolver, https://arxiv.org/abs/1304.3061) are the quantum-classic hybrid algorithms we mainly want to test.
# 
# ### Pseudoflow

# %%
algorithm = Pseudoflow()

# %% [markdown]
# ### Exact Diagonalization

# %%
algorithm = ExactDiagonalization()

# %% [markdown]
# ### ASP

# %%
algorithm = ASP(evol_time=10, nsteps=20)

# %% [markdown]
# Optionally, add a callback function, a callback frequency, and/or a specific backend to use

# %%
from qore.utils import get_bitstring_probabilities

# define a callback function
def analysis(circ, iter):
    print(f"--- Iter {iter} ---")
    x = get_bitstring_probabilities(circ, algorithm.quantum_instance)
    bitstr, prob = max(x.items(), key=lambda item: item[1])
    print(f"The most probable configuration and the corresponding probability: {bitstr, prob}")
    qmine.plot_mine_state(bitstr)


algorithm = ASP(
    evol_time=10,
    nsteps=20,
    callback=analysis,
    callback_freq=5,
    quantum_instance=QasmSimulator(),
)

# %% [markdown]
# ### QAOA

# %%
from qiskit.algorithms.optimizers import COBYLA

algorithm = QAOA(
        optimizer=COBYLA(),
        reps=3,
        quantum_instance=QasmSimulator(),
    )

# %% [markdown]
# ### VQE

# %%
from qiskit.circuit.library import EfficientSU2

algorithm = VQE(
        ansatz=EfficientSU2(
            qmine.nqubits,
            su2_gates=["ry"],
            entanglement="full",
            reps=3,
            insert_barriers=True,
        ),
        optimizer=COBYLA(),
        quantum_instance=QasmSimulator(),
    )

# %% [markdown]
# ### Now let's solve the problem with the selected algorithm

# %%
result = qmine.solve(algorithm)

# %% [markdown]
# ### Additional options
# 
# The second parameter in `Mine.solve()` can be a scalar or a `bool` or `None`. Default is `None`.
# 
# - When it evaluates to `False`, the projector Hamiltonian is used, which only allow valid (no constraint violation) states. This method does not introduce any new hyperparameters, but requires additional classical computation time. See the **ASP with projector** section for further explanation. 
# 
# - When it is a scalar, the penalty Hamiltonian is used and it serves as the penalty factor (a problem-dependent hyperparameter that needs to be chosen wisely).
# 
# - When it is `True`, an experimental heuristic penalty function (based on the mine values and number of sites) will be used to set the penalty. Warning: may be very inaccurate, especially for penalty-sensitive algorithms like QAOA and VQE).

# %%
result = qmine.solve(algorithm, penalty=11)

# %% [markdown]
# The third parameter in `Mine.solve()` can be used to enable the benchmark and profiling functions.
# 
# Setting it to `True` is equivalent to using the default settings of `qore.Benchmark()`.

# %%
result = qmine.solve(algorithm, benchmark=True)

# %% [markdown]
# Setting the `time_profiler` in `qore.Benchmark()` to "cProfile" to get more (but less readable) information. 
# 
# Toggling `profile_time` and `profile_memory` on or off to enable or disable the time and memory profilers when not needed. This may come in handy when we only want a wall time and want to avoid the additional overheads.
# 
# Setting the `path` to save the benchmark results in a json file.

# %%
from qore import Benchmark

result = qmine.solve(algorithm,
        benchmark=Benchmark(
            time_profiler="cProfile",
            profile_memory=False
        )
    )

# %% [markdown]
# ## ASP with Projector
# %% [markdown]
# #### Constructing projector $P$
# The projector $P$ is defined as:
# 
# $P|\psi\rangle = |\psi_{v}\rangle$,
# $P^{2}=P$,
# 
# where $|\psi\rangle$ is any state in the problem Hilbert space, and $|\psi_{v}\rangle$ lives in the subspace $V$ that obeys the mining constriants. Applying $I-H_{s}$ to a plus product state
# 
# $(I-H_{s})\prod |+\rangle = \sum_{i\in V}c_{i}|i\rangle$
# 
# results in a superposition state of all computational basis $|i\rangle$ that belong to $V$. Then the projector is constructed as
# 
# $P=\sum_{i\in V}|i\rangle\langle i|$
# 
# ### Incorporate with the eigensolvers
# The optimization problem is transformed to
# 
# $\min_{\psi} \frac{\langle \psi|PH_{p}P|\psi\rangle}{\langle \psi|P|\psi\rangle}$.
# 
# Note that $P$ is not unitary, and $\langle\psi(\theta)|P|\psi(\theta)\rangle \not= 1$ even for an ansatz that preserves the normality of the states. Rigorously speaking, an appropriate optimization algorithm for this problem should take the denominator $\langle P\rangle$ into account too. This is not immediately a ground-state-finding problem, and requires changing the qiskit VQE/QAOA source code to achieve. Luckily for the mining problem, the Hamiltonian is already diagonal in the computational basis, and the ground state is simply a product state. Hence treating the ground-state-finding problem
# 
# $\min_{\psi} \frac{\langle \psi|PH_{p}P|\psi\rangle}{\langle \psi|\psi\rangle}$
# 
# is adequate for our purpose, as long as the optimal mining configuration has a positive profit.

# %%
result = qmine.solve(ASP(10, 20))

