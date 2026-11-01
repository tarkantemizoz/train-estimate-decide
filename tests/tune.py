"""Tune BS and CBS on fewer, shorter trajectories, for the memory test in tests/test_models.py.

    python tests/peak_memory.py tests/tune.py <instance> <trajectories>

The levels are printed, not stored.
"""

import sys

from envs.lost_sales import cases
from envs.lost_sales.cases import tune

tune.SETTINGS |= {"number_of_trajectories": int(sys.argv[2]), "horizon": 100}
print(tune.tune(cases.find(sys.argv[1])))
