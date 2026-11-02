"""Stationary lost-sales super-MDP: the TED workshop environment.

    python train.py lost_sales/zero_shot_stationary demo

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
# An agent applies to every instance within the training ranges of its config.
SUPER_MDP = True
# Model dimensions when evaluating heuristics without a trained network: the model defaults.
DIMENSIONS = {}

__all__ = [
    "CASES",
    "DEFAULT_CASES",
    "DIMENSIONS",
    "MDP",
    "MODES",
    "SUPER_MDP",
    "BasePolicy",
    "SweepingBasePolicy",
    "Featurizer",
    "instance_config",
]
