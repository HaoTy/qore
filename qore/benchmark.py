from time import time, perf_counter
from typing import Any, Optional, Union


class Benchmark:
    def __init__(
        self,
        name: Optional[str] = None,
        path: Optional[str] = None,
        printout: Optional[bool] = True,
        reps: Optional[int] = 1,
        profile_time: Optional[bool] = True,
        time_profiler: Optional[str] = "pyinstrument",
        profile_memory: Optional[bool] = True,
        activate: Optional[bool] = True,
        additional_info: Optional[dict] = {},
    ) -> None:
        self.activate = activate
        self.name = name
        self.path = path
        self.printout = printout
        self.reps = reps
        self.profile_time = profile_time
        self.profile_memory = profile_memory
        self.time_profiler = time_profiler.lower()
        self.data = {}
        if name is not None:
            self.update_data("name", name)
        self.update_data(additional_info)
        self.result = None

    def __enter__(self) -> None:
        if self.activate:
            if self.profile_memory:
                import tracemalloc

                tracemalloc.start()

            if self.profile_time:
                if self.time_profiler == "cprofile":
                    import cProfile

                    self.profiler = cProfile.Profile()
                    self.profiler.enable()
                elif self.time_profiler == "pyinstrument":
                    from pyinstrument import Profiler

                    self.profiler = Profiler()
                    self.profiler.start()
            self.start_time = perf_counter()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.activate:
            duration = perf_counter() - self.start_time
            if self.profile_time:
                if self.time_profiler == "cprofile":
                    self.profiler.disable()

                    if self.printout:
                        import pstats
                        from io import StringIO

                        s = StringIO()
                        ps = pstats.Stats(self.profiler, stream=s)
                        ps.sort_stats("cumtime").print_stats(25)
                        ps.sort_stats("tottime").print_stats(25)
                        print(s.getvalue(), flush=True)

                elif self.time_profiler == "pyinstrument":
                    self.profiler.stop()
                    if self.printout:
                        print(self.profiler.output_text(unicode=True, color=True))

            if self.profile_memory:
                import tracemalloc

                _, peak_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.update_data("peak_memory", peak_memory)

            self.update_data("time", duration / self.reps)

            if self.result:
                self.update_data("result", self.result)

            try:
                import subprocess

                self.update_data(
                    "git_hash",
                    subprocess.check_output(["git", "describe", "--always"])
                    .strip()
                    .decode(),
                )
            except:
                pass

            self.save_data()

    def update_data(self, data: Union[dict, str], value: Optional[Any] = None) -> dict:
        if not isinstance(data, dict):
            data = {data: value}
        if self.printout:
            for k, v in data.items():
                print(f"{k}: {v}", flush=True)
        self.data.update(data)
        return self.data

    def save_data(self) -> dict:
        if self.path:
            import json

            try:
                with open(self.path, "r") as fd:
                    try:
                        database = json.load(fd)
                    except json.decoder.JSONDecodeError:
                        database = []
            except FileNotFoundError:
                database = []
            with open(self.path, "w+") as fd:
                database.append(self.data)
                json.dump(database, fd, indent=4)
        return self.data
