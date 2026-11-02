"""Heuristic policies: DCL's starting policy and the paper's known-parameter benchmarks.

When the model estimates, the policies evaluate.py runs free DynaPlex's garbage every period (gc()): the
estimates allocate every period, and DynaPlex 1.14 frees a heuristic's garbage only between trajectories,
so one estimated trajectory of 2000 periods held up to 330 MiB per worker. With known parameters a sweep
every period only costs time: it made tuning BS and CBS four times slower. DCL's starting policy has no
such safe point: in the code DCL compiles, one slows training by a third.
"""

from dynaplex import policy
from dynaplex.modelling import const_dataclass, gc

from .model import State, ZeroShotLostSalesMDP


@policy
@const_dataclass(slots=True)
class GreedyCappedBaseStockPolicy:
    """Order the most the action mask allows; DCL's starting policy."""

    mdp: ZeroShotLostSalesMDP

    def get_action(self, state: State) -> int:
        return state.order_constraint


@policy
@const_dataclass(slots=True)
class SweepingGreedyCappedBaseStockPolicy:
    """The greedy policy as evaluate.py runs it."""

    mdp: ZeroShotLostSalesMDP

    def get_action(self, state: State) -> int:
        if self.mdp.censored_demand or self.mdp.censored_leadtime:
            gc()
        return state.order_constraint


@policy
@const_dataclass(slots=True)
class BaseStockPolicy:
    mdp: ZeroShotLostSalesMDP
    base_stock_level: int

    def __post_init__(self):
        if not self.mdp.oracle_actions or not 0 <= self.base_stock_level < self.mdp.num_actions:
            raise ValueError("BaseStockPolicy needs legacy_oracle mode and 0 <= level < num_actions")

    def get_action(self, state: State) -> int:
        if self.mdp.censored_demand or self.mdp.censored_leadtime:
            gc()
        return max(0, self.base_stock_level - state.total_inv)


@policy
@const_dataclass(slots=True)
class CappedBaseStockPolicy:
    mdp: ZeroShotLostSalesMDP
    base_stock_level: int
    cap: int

    def __post_init__(self):
        if (
            not self.mdp.oracle_actions
            or self.base_stock_level < 0
            or not 0 <= self.cap < self.mdp.num_actions
        ):
            raise ValueError(
                "CappedBaseStockPolicy needs legacy_oracle mode, level >= 0, and 0 <= cap < num_actions"
            )

    def get_action(self, state: State) -> int:
        if self.mdp.censored_demand or self.mdp.censored_leadtime:
            gc()
        return max(0, min(self.cap, self.base_stock_level - state.total_inv))
