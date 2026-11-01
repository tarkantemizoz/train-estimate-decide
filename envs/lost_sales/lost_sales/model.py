"""One lost-sales instance with known parameters: the DynaML port of the legacy lost_sales_general model.

Demand is stationary or cyclic, and lead times fixed or stochastic, with or without order crossover.
A DCL agent trained on this model applies to its own instance only.
With the same seed it simulates exactly like zero_shot_lost_sales_large on the instance with known parameters,
including its crossover arrivals: one random number per age in transit, not per unit as in legacy.
"""

from dataclasses import dataclass
from typing import Final

from dynaplex import mdp
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

from ..cases import find
from ..cases.instance import fit_aer
from ..crossover import arriving_units


@dataclass(slots=True)
class State:
    state_vector: FifoQueue  # on-hand inventory first, then the orders in transit, newest last
    total_inv: int
    order_constraint: int
    arrival_draws: list[float]  # a period's crossover draws, one per age (see modify_state_with_event)
    period: int = 0  # position in the demand cycle
    category: StateCategory = StateCategory.AWAIT_ACTION


@trajectory_context
@dataclass
class LostSalesContext(TrajectoryContext):
    total_demand: int
    lost_units: int
    units_sold: int
    true_cost: float

    def __init__(self, model: "LostSalesMDP") -> None:
        super().__init__(model)
        self.total_demand = 0
        self.lost_units = 0
        self.units_sold = 0
        self.true_cost = 0.0


@mdp
@const_dataclass(init=False, slots=True)
class LostSalesMDP:
    p: float
    h: float
    demand: list[DiscreteDist]  # one distribution per period of the demand cycle
    cycle_length: int
    min_leadtime: int
    max_leadtime: int
    pipeline_width: int
    stochastic_leadtimes: bool
    order_crossover: bool
    arrival_thresholds: list[float]
    cycle_max_order_size: list[int]
    cycle_max_system_inv: list[int]
    max_order_size: int
    num_actions: int
    horizon_type: HorizonType

    def __init__(
        self,
        *,
        p: float,
        demand_cycles: list[int],
        mean_demand: list[float],
        std_demand: list[float],
        leadtime: int | None = None,
        stochastic_leadtime: bool = False,
        leadtime_distribution: list[float] | None = None,
        order_crossover: bool = False,
    ):
        if not p > 0:
            raise ValueError("p must be positive")
        if not len(demand_cycles) == len(mean_demand) == len(std_demand) >= 1:
            raise ValueError("demand_cycles, mean_demand and std_demand need one entry per period")
        if stochastic_leadtime:
            if leadtime is not None or leadtime_distribution is None:
                raise ValueError("Stochastic lead times need a leadtime_distribution and no fixed leadtime")
            probs = [float(x) for x in leadtime_distribution]
            if any(x < 0 for x in probs) or abs(sum(probs) - 1) >= 1e-8:
                raise ValueError("Lead-time probabilities must be nonnegative and sum to 1")
        else:
            if (
                type(leadtime) is not int
                or leadtime < 0
                or leadtime_distribution is not None
                or order_crossover
            ):
                raise ValueError(
                    "A fixed leadtime must be a nonnegative integer, without distribution or crossover"
                )
            probs = [0.0] * leadtime + [1.0]
        # Without crossover, an order also arrives when a later one does: legacy's arrival probabilities.
        arrival = probs
        if stochastic_leadtime and not order_crossover:
            arrival, cumulative, survival = [], 0.0, 1.0
            for prob in probs:
                cumulative += prob
                arrival.append(survival * max(0.0, cumulative))
                survival *= max(0.0, 1.0 - cumulative)
            if abs(sum(arrival) - 1) >= 1e-8:
                arrival = [x / sum(arrival) for x in arrival]
        low, high, found, total = 0, 0, False, 0.0
        for i, prob in enumerate(arrival):
            total += prob
            if not found and prob > 0.0:
                low, found = i, True
            if abs(total - 1.0) < 1e-8:
                high = i
                break

        self.p = float(p)
        self.h = 1.0
        self.demand = [fit_aer(float(m), float(s)) for m, s in zip(mean_demand, std_demand)]
        self.cycle_length = len(self.demand)
        self.min_leadtime = low
        self.max_leadtime = high
        self.pipeline_width = max(1, high)
        self.stochastic_leadtimes = low != high
        self.order_crossover = self.stochastic_leadtimes and order_crossover
        # An outstanding order arrives in a period when a uniform draw is at most its age's threshold.
        self.arrival_thresholds = [1.0] * len(arrival)
        if self.order_crossover:
            remaining = 1.0
            for i in range(high):
                self.arrival_thresholds[i] = probs[i] / remaining
                remaining -= probs[i]
        elif self.stochastic_leadtimes:
            cumulative = 0.0
            for i in range(high):
                cumulative += probs[i]
                self.arrival_thresholds[i] = cumulative
        # Newsvendor bounds per cycle period: demand at arrival caps the order, demand through arrival the system.
        ratio = self.p / (self.p + self.h)
        self.cycle_max_order_size = []
        self.cycle_max_system_inv = []
        for period in range(self.cycle_length):
            through = DiscreteDist.constant(0)
            at_arrival = DiscreteDist.constant(0)
            through_arrival = DiscreteDist.constant(0)
            weight = 0.0
            for age in range(high + 1):
                dist = self.demand[(period + age) % self.cycle_length]
                through = through.add(dist)
                prob = arrival[age]
                if age >= low and prob > 0.0:
                    if weight == 0.0:
                        at_arrival, through_arrival = dist, through
                    else:
                        at_arrival = at_arrival.mix(dist, prob / (weight + prob))
                        through_arrival = through_arrival.mix(through, prob / (weight + prob))
                    weight += prob
            self.cycle_max_order_size.append(at_arrival.fractile(ratio))
            self.cycle_max_system_inv.append(through_arrival.fractile(ratio))
        self.max_order_size = max(self.cycle_max_order_size)
        self.num_actions = self.max_order_size + 1
        self.horizon_type = HorizonType.INFINITE

    def make_context(self) -> LostSalesContext:
        return LostSalesContext(self)

    def get_initial_state(self, context: LostSalesContext) -> State:
        queue = FifoQueue()
        queue.reserve(self.pipeline_width + 1)
        queue.fill(self.pipeline_width)
        constraint = min(self.cycle_max_order_size[0], self.cycle_max_system_inv[0])
        draws = [0.0] * (self.max_leadtime if self.order_crossover else 0)
        return State(state_vector=queue, total_inv=0, order_constraint=constraint, arrival_draws=draws)

    def modify_state_with_action(self, state: State, context: LostSalesContext, action: int) -> None:
        assert state.category == StateCategory.AWAIT_ACTION, "Expected an action state"
        assert 0 <= action and action <= state.order_constraint, "Order exceeds action mask"
        state.state_vector.push_back(action)
        state.total_inv += action
        state.category = StateCategory.AWAIT_EVENT

    def modify_state_with_event(self, state: State, context: LostSalesContext) -> None:
        demand = self.demand[state.period].sample(context.rng)
        # Every arrival draw comes before any arrival, as in zero_shot_lost_sales_large: with crossover,
        # one per age that can arrive early, however many units it holds, and without, one in total.
        # Nothing is allocated per period: DynaPlex 1.14 does not count an empty list towards
        # garbage collection, so an empty list per period was never freed during DCL collection.
        draw = 0.0
        if self.order_crossover:
            if self.min_leadtime == 0:
                state.arrival_draws[0] = context.rng.random()
            for age in range(max(1, self.min_leadtime), self.max_leadtime):
                state.arrival_draws[age] = context.rng.random()
        elif self.stochastic_leadtimes:
            draw = context.rng.random()

        on_hand = state.state_vector.pop_front()
        arriving = 0
        if self.order_crossover:
            # Each unit in transit arrives independently, so orders may overtake each other.
            if self.min_leadtime == 0:
                location = len(state.state_vector) - 1
                received = arriving_units(
                    state.state_vector[location], self.arrival_thresholds[0], state.arrival_draws[0]
                )
                state.state_vector[location] -= received
                on_hand += received
            for age in range(max(1, self.min_leadtime), self.max_leadtime):
                location = self.pipeline_width - age
                received = arriving_units(
                    state.state_vector[location], self.arrival_thresholds[age], state.arrival_draws[age]
                )
                state.state_vector[location] -= received
                arriving += received
            location = self.pipeline_width - self.max_leadtime
            arriving += state.state_vector[location]
            state.state_vector[location] = 0
        elif self.stochastic_leadtimes:
            # The first arriving order brings every older one along, so orders keep their sequence.
            for age in range(self.min_leadtime, self.max_leadtime + 1):
                location = self.pipeline_width - max(1, age)
                count = state.state_vector[location]
                if draw <= self.arrival_thresholds[age] and count > 0:
                    if age == 0:
                        on_hand += count
                    else:
                        arriving += count
                    state.state_vector[location] = 0
                    for older in range(age + 1, self.max_leadtime + 1):
                        location = self.pipeline_width - older
                        arriving += state.state_vector[location]
                        state.state_vector[location] = 0
                    break
        else:
            location = self.pipeline_width - max(1, self.max_leadtime)
            if self.max_leadtime == 0:
                on_hand += state.state_vector[location]
            else:
                arriving += state.state_vector[location]
            state.state_vector[location] = 0

        sales = min(demand, on_hand)
        ending = on_hand - sales
        cost = self.h * ending + self.p * (demand - sales)
        context.true_cost += cost
        context.cumulative_cost += cost
        context.total_demand += demand
        context.units_sold += sales
        context.lost_units += demand - sales
        state.total_inv -= sales
        state.state_vector[0] = ending + arriving
        state.period = (state.period + 1) % self.cycle_length
        state.order_constraint = max(
            0,
            min(
                self.cycle_max_system_inv[state.period] - state.total_inv,
                self.cycle_max_order_size[state.period],
            ),
        )
        context.time_elapsed += 1
        state.category = StateCategory.AWAIT_ACTION

    def write_action_validity(self, state: State, valid: Validity) -> None:
        for action in range(self.num_actions):
            valid.set(action, action <= state.order_constraint)


@featurizer
@dataclass(slots=True)
class LostSalesFeaturizer(Featurizer):
    """The inventory pipeline, and the position in the demand cycle when demand is cyclic."""

    mdp: Final[LostSalesMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        for i in range(len(state.state_vector)):
            self.v.append(float(state.state_vector[i]))
        if self.mdp.cycle_length > 1:
            self.v.append(float(state.period))


def instance_config(instance, mode: str, dimensions: dict) -> dict:
    """The model of a case instance; its parameters are known, so only mode "known" applies."""
    if mode != "known":
        raise ValueError(f"A single-instance agent knows its parameters; mode {mode!r} does not apply")
    return dict(instance.parameters)


def mdp_config(instance_id: str) -> dict:
    """The model of the instance with this ID, such as online-003, for train.py --instance."""
    return instance_config(find(instance_id), "known", {})
