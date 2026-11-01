"""Cyclic lost-sales super-MDP with stochastic lead times: the paper's GC-LSN environment.

    python train.py zero_shot_lost_sales_large gc_lsn

train.py and evaluate.py use the names below.
"""

from .model import DIMENSIONS, instance_config
from .model import ZeroShotFeaturizer as Featurizer
from .model import ZeroShotLostSalesMDP as MDP
from .policies import GreedyCappedBaseStockPolicy as BasePolicy
from .policies import SweepingGreedyCappedBaseStockPolicy as SweepingBasePolicy

# The shared cases this model represents, the default selection, and its information modes.
CASES = ("case1", "case2", "case3", "online")
DEFAULT_CASES = CASES
MODES = ("known", "demand", "leadtime", "both")

__all__ = [
    "CASES",
    "DEFAULT_CASES",
    "DIMENSIONS",
    "MDP",
    "MODES",
    "BasePolicy",
    "SweepingBasePolicy",
    "Featurizer",
    "instance_config",
]
