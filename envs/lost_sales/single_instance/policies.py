"""Heuristic policies for the single-instance environment."""

from dynaplex import policy
from dynaplex.modelling import const_dataclass

from .model import LostSalesMDP, State


@policy
@const_dataclass(slots=True)
class GreedyCappedBaseStockPolicy:
    """Order the most the action mask allows; DCL's starting policy."""

    mdp: LostSalesMDP

    def get_action(self, state: State) -> int:
        return state.order_constraint
