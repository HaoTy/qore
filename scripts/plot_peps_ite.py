import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

reps = 3
sizes = list(range(2, 8))
algorithms = [
    "Pseudoflow 1st",
    "Pseudoflow 2nd",
    "PEPS Exact",
    # "PEPS BMPS d=1",
    "PEPS BMPS d=2",
    # "PEPS BMPS d=4",
    "PEPS BMPS no truncation",
]

sizes = np.array(sizes) * 2 - 1
timestamp = datetime.now()
results = np.load(f"./data/data-2022-03-27 18:12:30.625820.npy")[:,:,:-1]
markers = ["o", "^", "s", "x", "+", "D", "P", "*"]
fontsize = 24
markersize = 24
alpha = 0.75
linewidth = 3
markeredgewidth = 3

plt.figure(figsize=(9, 9))
for i, algo in enumerate(algorithms[2:]):
    result = results[i+2, 0] / reps
    result[result<0] = np.nan
    plt.semilogy(sizes, result, label=algo, marker=markers[i], markerfacecolor='none', markersize=markersize, alpha=alpha, linewidth=linewidth, markeredgewidth=markeredgewidth)

plt.xlabel("Mine Side Length", fontsize=fontsize)
plt.ylabel("Time (s)", fontsize=fontsize)
plt.xticks(sizes, fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize, loc="lower right")
plt.tight_layout()
plt.savefig(f"./plots/time-{timestamp}.png")
plt.close()

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.set_ylabel("Profit")
# ax2.set_ylabel("Violation")

# for i, algo in enumerate(algorithms):
#     ax1.plot(sizes, results[i, 1], label=algo, color="r", marker=markers[i])
#     ax2.plot(sizes, results[i, 2], label=algo, color="b", marker=markers[i])

# plt.legend()
# fig.savefig(f"./plots/profit.png")
# plt.close()

plt.figure(figsize=(9, 4))
for i, algo in enumerate(algorithms[2:]):
    result = results[i+2, 1] / results[1, 1]
    result[result<0] = np.nan
    plt.plot(sizes, result, label=algo, marker=markers[i], markerfacecolor='none', markersize=markersize, alpha=alpha, linewidth=linewidth, markeredgewidth=markeredgewidth)

plt.xlabel("Mine Side Length", fontsize=fontsize)
plt.ylabel("Normalized Profit", fontsize=fontsize)
plt.xticks(sizes, fontsize=fontsize)
plt.yticks([0.5, 1.0, 1.5], fontsize=fontsize)
plt.ylim([0.5, 1.5])
plt.tight_layout()
plt.savefig(f"./plots/profit-{timestamp}.png")
plt.close()

plt.figure(figsize=(9, 4))
for i, algo in enumerate(algorithms[2:]):
    result = results[i+2, 2] / reps
    result[result<0] = np.nan
    plt.plot(sizes, result, label=algo, marker=markers[i], markerfacecolor='none', markersize=markersize, alpha=alpha, linewidth=linewidth, markeredgewidth=markeredgewidth)

plt.xlabel("Mine Side Length", fontsize=fontsize)
plt.ylabel("Violations", fontsize=fontsize)
plt.xticks(sizes, fontsize=fontsize)
plt.yticks([-1, 0, 1], fontsize=fontsize)
plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig(f"./plots/violation-{timestamp}.png")
plt.close()
