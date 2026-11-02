"""Lost sales on one instance with known parameters: DCL trains an agent for that instance only.

    python train.py lost_sales/single_instance demo
    python train.py lost_sales/single_instance demo --instance online-003

A config's "mdp" section holds the parameters of its instance: demo trains online-012.
train.py and evaluate.py use the names below; mdp_config gives the parameters of the case instance --instance names.
"""

from .model import LostSalesFeaturizer as Featurizer
from .model import LostSalesMDP as MDP
from .model import instance_config, mdp_config
from .policies import GreedyCappedBaseStockPolicy as BasePolicy

# The shared cases this model represents, the default selection, and its information modes.
CASES = ("case1", "case2", "case3", "online")
DEFAULT_CASES = ("online",)
MODES = ("known",)
# An agent applies to the instance in its config only, not to ranges of instances.
SUPER_MDP = False
# A single-instance model has no dimensions beyond its instance.
DIMENSIONS = {}
# The model allocates nothing per period, so evaluate.py needs no sweeping copy of the base policy.
SweepingBasePolicy = BasePolicy

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
    "mdp_config",
]
