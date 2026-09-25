"""Faithful port of the local legacy Zero_Shot_Lost_Sales_Stationary model.

The observation estimator deliberately preserves the legacy redistribution rule,
which differs from exact Kaplan-Meier and uses an excess-demand indicator.
The legacy_oracle action mode permits unclipped BS/CBS benchmark orders.
"""

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from dynaplex import mdp, policy
from dynaplex.modelling import (
    DiscreteDist,
    Featurizer,
    FifoQueue,
    GlobalStateWriter,
    HorizonType,
    StateCategory,
    TrajectoryContext,
    Validity,
    const_dataclass,
    featurizer,
    trajectory_context,
)
from numpy.typing import NDArray


@dataclass(slots=True)
class State:
    state_vector: FifoQueue
    total_inv: int
    leadtime: int
    p: float
    demand_dist: DiscreteDist
    mean_demand: float
    std_demand: float
    max_order_size: int
    max_system_inv: int
    max_order_size_limit: int
    order_constraint: int
    demand_probs: NDArray[np.float64]
    est_min_demand: int
    category: StateCategory = StateCategory.AWAIT_ACTION
    demand: int = 0
    collect_demand_statistics: bool = False
    period_count: int = 0
    past_demands: list[int] = field(default_factory=list)
    cumulative_demands: list[int] = field(default_factory=list)
    censor_indicator: list[int] = field(default_factory=list)


@trajectory_context
@dataclass
class LostSalesContext(TrajectoryContext):
    """Evaluation totals reset after warmup without resetting the estimator."""

    total_demand: int
    lost_units: int
    units_sold: int
    true_cost: float

    def __init__(self, model: "StationaryLostSalesMDP") -> None:
        super().__init__(model)
        self.total_demand = 0
        self.lost_units = 0
        self.units_sold = 0
        self.true_cost = 0.0


@mdp
@const_dataclass(init=False, slots=True)
class StationaryLostSalesMDP:
    evaluate: bool
    censored_demand: bool
    oracle_actions: bool
    max_p: float
    max_demand: float
    max_leadtime: int
    pipeline_width: int
    p: float
    leadtime: int
    mean_demand: float
    std_demand: float
    h: float
    max_order_size: int
    max_system_inv: int
    num_actions: int
    horizon_type: HorizonType

    def __init__(
        self,
        *,
        evaluate: bool = False,
        max_p: float = 100.0,
        max_demand: float = 12.0,
        max_leadtime: int = 10,
        p: float | None = None,
        leadtime: int | None = None,
        mean_demand: float | None = None,
        std_demand: float | None = None,
        censored_demand: bool = False,
        action_mode: str = "masked",
        discount_factor: float = 1.0,
    ):
        if not math.isfinite(max_p) or max_p < 2.0:
            raise ValueError("max_p must be finite and >= 2")
        if not math.isfinite(max_demand) or max_demand < 2.0:
            raise ValueError("max_demand must be finite and >= 2")
        if type(max_leadtime) is not int or max_leadtime < 0:
            raise ValueError("max_leadtime must be a nonnegative integer")
        if discount_factor != 1.0:
            raise ValueError("This port supports average cost (discount_factor=1) only")
        if censored_demand and not evaluate:
            raise ValueError("censored_demand is supported only in evaluate mode")
        if action_mode not in ("masked", "legacy_oracle"):
            raise ValueError("action_mode must be 'masked' or 'legacy_oracle'")
        if action_mode == "legacy_oracle" and not evaluate:
            raise ValueError("legacy_oracle actions are for benchmark evaluation only")
        if evaluate:
            if p is None or not math.isfinite(p) or p <= 0:
                raise ValueError("evaluate mode requires finite p > 0")
            if type(leadtime) is not int or not 0 <= leadtime <= max_leadtime:
                raise ValueError("evaluate mode requires 0 <= leadtime <= max_leadtime")
            if mean_demand is None or not math.isfinite(mean_demand) or mean_demand <= 0:
                raise ValueError("evaluate mode requires finite mean_demand > 0")
            if std_demand is None or not math.isfinite(std_demand) or std_demand < 0:
                raise ValueError("evaluate mode requires finite std_demand >= 0")
            # Validate the requested moment fit before entering a compiled trajectory.
            DiscreteDist.adan_eenige_resing(float(mean_demand), float(std_demand))
        elif any(x is not None for x in (p, leadtime, mean_demand, std_demand)):
            raise ValueError("Fixed instance parameters require evaluate=True")

        self.evaluate = evaluate
        self.censored_demand = censored_demand
        self.oracle_actions = action_mode == "legacy_oracle"
        self.max_p = float(max_p)
        self.max_demand = float(max_demand)
        self.max_leadtime = max_leadtime
        # Legacy has max_leadtime entries at a decision epoch, including on-hand.
        # One entry also makes the previously broken max_leadtime=0 case well-defined.
        self.pipeline_width = max(1, max_leadtime)
        self.p = float(p) if p is not None else 2.0
        self.leadtime = leadtime if leadtime is not None else 0
        self.mean_demand = float(mean_demand) if mean_demand is not None else 2.0
        self.std_demand = float(std_demand) if std_demand is not None else 0.0
        self.h = 1.0
        dist = DiscreteDist.adan_eenige_resing(self.max_demand, 2.0 * self.max_demand)
        total = DiscreteDist.constant(0)
        for _ in range(max_leadtime + 1):
            total = total.add(dist)
        ratio = self.max_p / (self.max_p + self.h)
        self.max_order_size = dist.fractile(ratio)
        self.max_system_inv = total.fractile(ratio)
        self.num_actions = (self.max_system_inv if self.oracle_actions else self.max_order_size) + 1
        self.horizon_type = HorizonType.INFINITE

    def make_context(self) -> LostSalesContext:
        return LostSalesContext(self)

    def get_initial_state(self, context: LostSalesContext) -> State:
        penalty = self.p
        leadtime = self.leadtime
        mean = self.mean_demand
        stdev = self.std_demand
        if not self.evaluate:
            penalty = context.rng.random() * (self.max_p - 2.0) + 2.0
            leadtime = int(math.floor(context.rng.random() * (self.max_leadtime + 1)))
            mean = context.rng.random() * (self.max_demand - 2.0) + 2.0
            min_std = math.sqrt(DiscreteDist.least_variance_for_moment_fit(mean))
            stdev = context.rng.random() * (mean * 2.0 - min_std) + min_std
        dist = DiscreteDist.adan_eenige_resing(mean, stdev)
        estimate = dist
        if self.censored_demand:
            estimate = DiscreteDist.constant(int(math.ceil(self.max_demand)))
            mean = estimate.expectation()
            stdev = estimate.std()
        queue = FifoQueue()
        queue.reserve(self.pipeline_width + 1)
        queue.fill(self.pipeline_width)
        probs = np.zeros(estimate.max() - estimate.min() + 1, dtype=np.float64)
        for value in range(estimate.min(), estimate.max() + 1):
            probs[value - estimate.min()] = estimate.probability_at(value)
        state = State(
            state_vector=queue,
            total_inv=0,
            leadtime=leadtime,
            p=penalty,
            demand_dist=dist,
            mean_demand=mean,
            std_demand=stdev,
            max_order_size=0,
            max_system_inv=0,
            max_order_size_limit=self.max_order_size,
            order_constraint=0,
            demand_probs=probs,
            est_min_demand=estimate.min(),
        )
        self.update_order_limits(state)
        if not self.censored_demand:
            state.max_order_size_limit = state.max_order_size
        return state

    def update_order_limits(self, state: State) -> None:
        dist = DiscreteDist.custom(state.demand_probs, state.est_min_demand)
        total = DiscreteDist.constant(0)
        for _ in range(state.leadtime + 1):
            total = total.add(dist)
        ratio = state.p / (state.p + self.h)
        state.max_order_size = min(dist.fractile(ratio), state.max_order_size_limit)
        state.max_system_inv = min(total.fractile(ratio), self.max_system_inv)
        state.order_constraint = max(0, min(state.max_system_inv - state.total_inv, state.max_order_size))

    def modify_state_with_action(self, state: State, context: LostSalesContext, action: int) -> None:
        assert state.category == StateCategory.AWAIT_ACTION, "Expected an action state"
        assert action >= 0 and action < self.num_actions, "Order outside the configured action domain"
        assert self.oracle_actions or action <= state.order_constraint, "Order exceeds action mask"
        state.state_vector.push_back(action)
        state.total_inv += action
        state.category = StateCategory.AWAIT_EVENT

    def modify_state_with_event(self, state: State, context: LostSalesContext) -> None:
        self.apply_demand(state, context, state.demand_dist.sample(context.rng))

    def apply_demand(self, state: State, context: LostSalesContext, demand: int) -> None:
        """A period with an explicit demand, also usable for controlled replay."""
        assert state.category == StateCategory.AWAIT_EVENT, "Place an order before applying demand"
        assert demand >= 0, "Demand must be nonnegative"
        on_hand = state.state_vector.pop_front()
        location = self.pipeline_width - max(1, state.leadtime)
        arrival = state.state_vector[location]
        state.state_vector[location] = 0
        if state.leadtime == 0:
            on_hand += arrival
            arrival = 0
        if self.censored_demand and on_hand > 0:
            state.collect_demand_statistics = True
        state.demand = demand
        sales = min(on_hand, demand)
        lost = demand - sales
        ending = on_hand - sales
        true_cost = self.h * ending + state.p * lost
        context.true_cost += true_cost
        context.total_demand += demand
        context.lost_units += lost
        context.units_sold += sales
        if self.censored_demand:
            context.cumulative_cost += self.h * ending - state.p * sales
        else:
            context.cumulative_cost += true_cost
        state.total_inv -= sales
        state.state_vector[0] = ending + arrival
        if self.censored_demand:
            if state.collect_demand_statistics:
                # Equality is deliberately uncensored, as in the legacy code.
                self.update_demand_statistics(state, demand <= on_hand, sales)
            self.update_order_limits(state)
        state.order_constraint = max(0, min(state.max_system_inv - state.total_inv, state.max_order_size))
        context.time_elapsed += 1
        state.category = StateCategory.AWAIT_ACTION

    def update_demand_statistics(self, state: State, uncensored: bool, observation: int) -> None:
        """Legacy one-pass redistribution, including its terminal tail convention."""
        state.period_count += 1
        old_size = len(state.past_demands) - 1
        if observation > old_size:
            for _ in range(old_size + 1, observation):
                state.past_demands.append(0)
                state.censor_indicator.append(0)
                state.cumulative_demands.append(1)
            state.past_demands.append(1)
            state.cumulative_demands.append(0)
            state.censor_indicator.append(0 if uncensored else 1)
        else:
            state.past_demands[observation] += 1
            old_size = observation - 1
            if not uncensored:
                state.censor_indicator[observation] += 1
        for i in range(old_size + 1):
            state.cumulative_demands[i] += 1
        size = len(state.past_demands)
        probs = np.zeros(size, dtype=np.float64)
        for i in range(size):
            probs[i] += float(state.past_demands[i]) / state.period_count
            if state.censor_indicator[i] > 0 and i < size - 1:
                weight = float(state.censor_indicator[i]) / state.period_count
                probs[i] -= weight
                for j in range(i + 1, size):
                    probs[j] += state.past_demands[j] * weight / state.cumulative_demands[i]
        state.demand_probs = probs
        state.est_min_demand = 0
        dist = DiscreteDist.custom(probs, 0)
        state.mean_demand = dist.expectation()
        state.std_demand = dist.std()

    def write_action_validity(self, state: State, valid: Validity) -> None:
        for action in range(self.num_actions):
            valid.set(action, self.oracle_actions or action <= state.order_constraint)


@featurizer
@dataclass(slots=True)
class StationaryFeaturizer(Featurizer):
    mdp: Final[StationaryLostSalesMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        self.v.append(state.p)
        for i in range(len(state.state_vector)):
            self.v.append(float(state.state_vector[i]))
        self.v.append(float(state.leadtime))
        self.v.append(state.mean_demand)
        self.v.append(state.std_demand)


@policy
@const_dataclass(slots=True)
class GreedyCappedBaseStockPolicy:
    mdp: StationaryLostSalesMDP

    def get_action(self, state: State) -> int:
        return state.order_constraint


@policy
@const_dataclass(slots=True)
class BaseStockPolicy:
    mdp: StationaryLostSalesMDP
    base_stock_level: int

    def __post_init__(self):
        if not self.mdp.oracle_actions or not 0 <= self.base_stock_level < self.mdp.num_actions:
            raise ValueError("BaseStockPolicy needs legacy_oracle mode and 0 <= level < num_actions")

    def get_action(self, state: State) -> int:
        return max(0, self.base_stock_level - state.total_inv)


@policy
@const_dataclass(slots=True)
class CappedBaseStockPolicy:
    mdp: StationaryLostSalesMDP
    base_stock_level: int
    cap: int

    def __post_init__(self):
        if (
            not self.mdp.oracle_actions
            or self.base_stock_level < 0
            or not 0 <= self.cap < self.mdp.num_actions
        ):
            raise ValueError(
                "CappedBaseStockPolicy needs legacy_oracle mode, level >= 0 and 0 <= cap < num_actions"
            )

    def get_action(self, state: State) -> int:
        return max(0, min(self.cap, self.base_stock_level - state.total_inv))


@policy
@const_dataclass(slots=True)
class ConstantOrderPolicy:
    mdp: StationaryLostSalesMDP
    order: int

    def __post_init__(self):
        if not self.mdp.oracle_actions or not 0 <= self.order < self.mdp.num_actions:
            raise ValueError("ConstantOrderPolicy needs legacy_oracle mode and 0 <= order < num_actions")

    def get_action(self, state: State) -> int:
        return self.order
