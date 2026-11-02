"""Run a Python script as if from the repository root, and print its peak memory (MiB) last.

    python tests/peak_memory.py evaluate.py lost_sales/zero_shot_stationary --modes demand
    python tests/peak_memory.py tests/collect.py lost_sales/single_instance 40

The memory tests compare a small and a large run, so that memory growing with the run shows.
"""

import resource
import runpy
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
# ru_maxrss is in bytes on macOS and in KiB on Linux.
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak / 2**20 if sys.platform == "darwin" else peak / 2**10)
