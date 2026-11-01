"""Heuristic policies for the stationary environment."""

from dynaplex import policy
from dynaplex.modelling import const_dataclass, gc

from .model import State, StationaryLostSalesMDP


@policy
@const_dataclass(slots=True)
class GreedyCappedBaseStockPolicy:
    """Order the most the action mask allows; DCL's starting policy."""

    mdp: StationaryLostSalesMDP

    def get_action(self, state: State) -> int:
        return state.order_constraint


@policy
@const_dataclass(slots=True)
class SweepingGreedyCappedBaseStockPolicy:
    """The greedy policy as evaluate.py runs it: when the model estimates, it also frees DynaPlex's garbage.

    The demand estimate allocates every period, and DynaPlex 1.14 frees a heuristic's garbage only between
    trajectories, so one estimated trajectory of 2000 periods held about 170 MiB per worker. With known
    demand a sweep every period only costs time. DCL's starting policy has no such safe point: in the code
    DCL compiles, one slows training by a third.
    """

    mdp: StationaryLostSalesMDP

    def get_action(self, state: State) -> int:
        if self.mdp.censored_demand:
            gc()
        return state.order_constraint
