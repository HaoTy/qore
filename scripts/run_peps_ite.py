from koala.peps.contraction import Snake
import numpy as np
from qore import Mine, PEPSITE, Benchmark, Pseudoflow, ASP
from tensorbackends.interface import ReducedSVD, RandomizedSVD, ImplicitRandomizedSVD
from koala import peps
from datetime import datetime


if __name__ == "__main__":
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
    results = np.zeros((len(algorithms), 3, len(sizes)))

    for i, m in enumerate(sizes):
        # evol_time = 3 - m / 5
        evol_time = 6
        for k in range(reps):
            qmine = Mine.gen_random_mine((m, 2 * m - 1), loc=0.1, scale=1)
            # qmine.plot_mine()

            for j, algo in enumerate(
                (
                    Pseudoflow(),
                    Pseudoflow(),
                    # ASP(10, 20),
                    PEPSITE(evol_time, Snake()),
                    # PEPSITE(evol_time, peps.BMPS(ImplicitRandomizedSVD(1))),
                    PEPSITE(evol_time, peps.BMPS(ImplicitRandomizedSVD(2))),
                    # PEPSITE(evol_time, peps.BMPS(ImplicitRandomizedSVD(4))),
                    PEPSITE(evol_time, peps.BMPS()),
                )
            ):
                if (i > 2 and j == 4) or (i > 4 and j == 2):
                    results[j, 0, i] = -1
                    results[j, 1, i] = -1
                    results[j, 2, i] = -1
                    continue
                bench = Benchmark(f"{i}-{algorithms[j]}", profile_time=False, profile_memory=False)
                with bench:
                    bitstr = qmine.solve(algo).optimal_config
                results[j, 0, i] += bench.data["time"]
                results[j, 1, i] += qmine.get_profit(bitstr)
                results[j, 2, i] += qmine.get_violation(bitstr)

    np.save(f"./data/data-{datetime.now()}", results)
