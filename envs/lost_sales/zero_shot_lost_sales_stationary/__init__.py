"""Stationary lost-sales super-MDP: the TED workshop environment.

    python train.py zero_shot_lost_sales_stationary demo

train.py and evaluate.py use the names below.
"""

from .model import StationaryFeaturizer as Featurizer
from .model import StationaryLostSalesMDP as MDP
from .model import instance_config
from .policies import GreedyCappedBaseStockPolicy as BasePolicy
from .policies import SweepingGreedyCappedBaseStockPolicy as SweepingBasePolicy

# The shared cases this model represents, the default selection, and its information modes.
CASES = ("case1", "online")
DEFAULT_CASES = ("online",)
MODES = ("known", "demand")
# Model dimensions when evaluating heuristics without a trained network: the model defaults.
DIMENSIONS = {}

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
