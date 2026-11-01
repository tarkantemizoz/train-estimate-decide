"""DynaML port of the legacy cyclic, stochastic-lead-time lost-sales super-MDP.

Unlike legacy, crossover arrivals take one random number per age in transit, not per unit (see crossover.py),
so that every policy draws the same numbers whatever it orders.
"""

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
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

from ..cases.instance import fit_aer
from ..crossover import arriving_units


@dataclass(slots=True)
class DemandPeriod:
    group: int
    true_dist: DiscreteDist
    estimate: DiscreteDist
    mean: float
    std: float
    collect: bool = False
    count: int = 0
    past: list[int] = field(default_factory=list)
    cumulative: list[int] = field(default_factory=list)
    censored: list[int] = field(default_factory=list)


@dataclass(slots=True)
class State:
    state_vector: FifoQueue
    periods: list[DemandPeriod]
    p: float
    min_leadtime: int
    max_leadtime: int
    estimated_min_leadtime: int
    estimated_max_leadtime: int
    estimated_leadtime_probs: list[float]
    arrival_thresholds: list[float]
    stochastic_leadtimes: bool
    order_crossover: bool
    past_leadtimes: list[int]
    cycle_max_order_size: list[int]
    cycle_max_system_inv: list[int]
    arrival_draws: list[float]  # a period's crossover draws, one per age (see modify_state_with_event)
    total_inv: int = 0
    period: int = 0
    demand: int = 0
    orders_received: int = 0
    max_order_size: int = 0
    max_system_inv: int = 0
    max_order_size_limit: int = 0
    order_constraint: int = 0
    category: StateCategory = StateCategory.AWAIT_ACTION


@trajectory_context
@dataclass
class LostSalesContext(TrajectoryContext):
    total_demand: int
    lost_units: int
    units_sold: int
    true_cost: float

    def __init__(self, model: "ZeroShotLostSalesMDP") -> None:
        super().__init__(model)
        self.total_demand = 0
        self.lost_units = 0
        self.units_sold = 0
        self.true_cost = 0.0


@mdp
@const_dataclass(init=False, slots=True)
class ZeroShotLostSalesMDP:
    evaluate: bool
    train_stochastic_leadtimes: bool
    train_cyclic_demand: bool
    censored_demand: bool
    censored_leadtime: bool
    maximize_rewards: bool
    order_crossover: bool
    oracle_actions: bool
    max_p: float
    max_demand: float
    max_leadtime: int
    max_num_cycles: int
    pipeline_width: int
    p: float
    demand_cycles: list[int]
    mean_demand: list[float]
    std_demand: list[float]
    leadtime_probs: list[float]
    non_crossing_probs: list[float]
    max_order_size: int
    max_system_inv: int
    num_actions: int
    num_features: int
    horizon_type: HorizonType

    def __init__(
        self,
        *,
        evaluate: bool = False,
        train_stochastic_leadtimes: bool = True,
        train_cyclic_demand: bool = True,
        max_p: float = 100.0,
        max_demand: float = 12.0,
        max_leadtime: int = 10,
        max_num_cycles: int = 7,
        p: float | None = None,
        demand_cycles: list[int] | None = None,
        mean_demand: list[float] | None = None,
        std_demand: list[float] | None = None,
        leadtime: int | None = None,
        stochastic_leadtime: bool = False,
        leadtime_distribution: list[float] | None = None,
        order_crossover: bool = False,
        censored_demand: bool = False,
        censored_leadtime: bool = False,
        maximize_rewards: bool = False,
        action_mode: str = "masked",
        discount_factor: float = 1.0,
    ):
        if not math.isfinite(max_p) or max_p < 2:
            raise ValueError("max_p must be finite and >= 2")
        if not math.isfinite(max_demand) or max_demand < 2:
            raise ValueError("max_demand must be finite and >= 2")
        if type(max_leadtime) is not int or max_leadtime < 0:
            raise ValueError("max_leadtime must be a nonnegative integer")
        if type(max_num_cycles) is not int or max_num_cycles < 1:
            raise ValueError("max_num_cycles must be a positive integer")
        if discount_factor != 1.0:
            raise ValueError("Only infinite-horizon average cost (discount_factor=1) is supported")
        if action_mode not in ("masked", "legacy_oracle"):
            raise ValueError("action_mode must be 'masked' or 'legacy_oracle'")
        if not evaluate and (
            any(
                x is not None
                for x in (p, demand_cycles, mean_demand, std_demand, leadtime, leadtime_distribution)
            )
            or stochastic_leadtime
            or order_crossover
            or censored_demand
            or censored_leadtime
            or maximize_rewards
            or action_mode != "masked"
        ):
            raise ValueError("Fixed instance parameters, censoring, and oracle actions require evaluate=True")
        self.evaluate = evaluate
        self.train_stochastic_leadtimes = train_stochastic_leadtimes
        self.train_cyclic_demand = train_cyclic_demand
        self.censored_demand = censored_demand
        self.censored_leadtime = censored_leadtime
        self.maximize_rewards = maximize_rewards
        self.order_crossover = order_crossover
        self.oracle_actions = action_mode == "legacy_oracle"
        self.max_p = float(max_p)
        self.max_demand = float(max_demand)
        self.max_leadtime = max_leadtime
        self.max_num_cycles = max_num_cycles
        self.pipeline_width = max(1, max_leadtime)
        self.p = float(p) if p is not None else 2.0
        self.demand_cycles = list(demand_cycles) if demand_cycles is not None else [0]
        self.mean_demand = [float(x) for x in mean_demand] if mean_demand is not None else [2.0]
        self.std_demand = [float(x) for x in std_demand] if std_demand is not None else [0.0]
        self.leadtime_probs = [0.0] * (max_leadtime + 1)
        self.non_crossing_probs = [0.0] * (max_leadtime + 1)
        if evaluate:
            if p is None or not math.isfinite(p) or p <= 0:
                raise ValueError("evaluate mode requires finite p > 0")
            if any(x is None for x in (demand_cycles, mean_demand, std_demand)):
                raise ValueError("evaluate mode requires demand_cycles, mean_demand, and std_demand")
            size = len(self.demand_cycles)
            if (
                not 1 <= size <= max_num_cycles
                or len(self.mean_demand) != size
                or len(self.std_demand) != size
            ):
                raise ValueError("Demand vectors must have equal lengths between 1 and max_num_cycles")
            if not train_cyclic_demand and size != 1:
                raise ValueError("Cyclic evaluation requires train_cyclic_demand=True for complete features")
            for i, group in enumerate(self.demand_cycles):
                if type(group) is not int or not 0 <= group <= i or self.demand_cycles[group] != group:
                    raise ValueError("demand_cycles labels must refer to their first period index")
                mean, std = self.mean_demand[i], self.std_demand[i]
                if not math.isfinite(mean) or mean <= 0 or not math.isfinite(std) or std < 0:
                    raise ValueError(
                        "Demand means must be finite and positive; standard deviations nonnegative"
                    )
                if (mean, std) != (self.mean_demand[group], self.std_demand[group]):
                    raise ValueError("Periods sharing a demand group must have the same true distribution")
                fit_aer(mean, std)
            if stochastic_leadtime:
                if leadtime is not None:
                    raise ValueError("Choose leadtime or a stochastic leadtime_distribution, not both")
                if not train_stochastic_leadtimes:
                    raise ValueError("Stochastic evaluation requires train_stochastic_leadtimes=True")
                if leadtime_distribution is None or len(leadtime_distribution) != max_leadtime + 1:
                    raise ValueError("leadtime_distribution must have max_leadtime + 1 entries")
                probs = [float(x) for x in leadtime_distribution]
                if any(not math.isfinite(x) or x < 0 for x in probs) or abs(sum(probs) - 1) >= 1e-8:
                    raise ValueError("Lead-time probabilities must be finite, nonnegative, and sum to 1")
                self.leadtime_probs = probs
                if not order_crossover:
                    self.non_crossing_probs = probs
                    cumulative, survival = 0.0, 1.0
                    marginal = []
                    for prob in probs:
                        cumulative += prob
                        marginal.append(survival * max(0.0, cumulative))
                        survival *= max(0.0, 1.0 - cumulative)
                    if abs(sum(marginal) - 1) >= 1e-8:
                        marginal = [x / sum(marginal) for x in marginal]
                    self.leadtime_probs = marginal
            else:
                if type(leadtime) is not int or not 0 <= leadtime <= max_leadtime:
                    raise ValueError("Deterministic evaluation requires 0 <= leadtime <= max_leadtime")
                if leadtime_distribution is not None or order_crossover or censored_leadtime:
                    raise ValueError(
                        "Lead-time distributions, crossover, and lead-time estimation require stochastic_leadtime"
                    )
                self.leadtime_probs[leadtime] = 1.0
        dist = fit_aer(self.max_demand, 2.0 * self.max_demand)
        total = DiscreteDist.constant(0)
        for _ in range(max_leadtime + 1):
            total = total.add(dist)
        ratio = self.max_p / (self.max_p + 1.0)
        self.max_order_size = dist.fractile(ratio)
        self.max_system_inv = total.fractile(ratio)
        self.num_actions = (self.max_system_inv if self.oracle_actions else self.max_order_size) + 1
        self.num_features = 1 + self.pipeline_width
        self.num_features += max_leadtime + 2 if train_stochastic_leadtimes else 1
        self.num_features += 1 + 2 * max_num_cycles if train_cyclic_demand else 2
        self.horizon_type = HorizonType.INFINITE

    def make_context(self) -> LostSalesContext:
        return LostSalesContext(self)

    def sample_leadtime_distribution(self, context: LostSalesContext, low: int, high: int) -> list[float]:
        probs: list[float] = []
        for _ in range(self.max_leadtime + 1):
            probs.append(0.0)
        if low == high:
            probs[low] = 1.0
        elif context.rng.random() < 0.33:
            for i in range(low, high + 1):
                probs[i] = 1.0 / (high - low + 1)
        else:
            mean = float(high + low) / 2.0
            if context.rng.random() < 0.5 and mean > 2.0:
                min_std = math.sqrt(DiscreteDist.least_variance_for_moment_fit(mean))
                std = context.rng.random() * (mean * 2.0 - min_std) + min_std
                dist = fit_aer(mean, std)
                for i in range(low, high + 1):
                    probs[i] = dist.probability_at(i)
            else:
                remaining = 1.0
                for i in range(low, high):
                    probs[i] = max(0.0, context.rng.random() * remaining)
                    remaining -= probs[i]
                probs[high] = max(0.0, remaining)
        total = sum(probs)
        assert total > 0.0, "Lead-time fit has no mass on requested support"
        if abs(total - 1.0) >= 1e-8:
            for i in range(len(probs)):
                probs[i] /= total
        return probs

    def get_initial_state(self, context: LostSalesContext) -> State:
        penalty = self.p
        low = 0
        high = 0
        size = len(self.demand_cycles)
        lt_probs: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        groups: list[int] = []
        if self.evaluate:
            for x in self.leadtime_probs:
                lt_probs.append(x)
            for x in self.mean_demand:
                means.append(x)
            for x in self.std_demand:
                stds.append(x)
            for x in self.demand_cycles:
                groups.append(x)
            found = False
            total = 0.0
            for i in range(self.max_leadtime + 1):
                total += lt_probs[i]
                if not found and lt_probs[i] > 0.0:
                    low = i
                    found = True
                if abs(total - 1.0) < 1e-8:
                    high = i
                    break
        else:
            penalty = context.rng.random() * (self.max_p - 2.0) + 2.0
            low = int(math.floor(context.rng.random() * (self.max_leadtime + 1)))
            high = low
            if self.train_stochastic_leadtimes:
                high = low + int(math.floor(context.rng.random() * (self.max_leadtime - low + 1)))
            lt_probs = self.sample_leadtime_distribution(context, low, high)
            size = 1
            if self.train_cyclic_demand:
                size += int(math.floor(context.rng.random() * self.max_num_cycles))
            for i in range(size):
                groups.append(i)
                mean = context.rng.random() * (self.max_demand - 2.0) + 2.0
                min_std = math.sqrt(DiscreteDist.least_variance_for_moment_fit(mean))
                means.append(mean)
                stds.append(context.rng.random() * (mean * 2.0 - min_std) + min_std)
        stochastic = low != high
        crossover = False
        thresholds: list[float] = []
        estimates: list[float] = []
        past_leadtimes: list[int] = []
        for x in lt_probs:
            thresholds.append(1.0)
            estimates.append(x)
            past_leadtimes.append(0)
        if stochastic:
            crossover = self.order_crossover if self.evaluate else context.rng.random() >= 0.5
            if crossover:
                remaining = 1.0
                for i in range(high):
                    thresholds[i] = lt_probs[i] / remaining
                    remaining -= lt_probs[i]
            else:
                cumulative = 0.0
                for i in range(high):
                    cumulative += self.non_crossing_probs[i] if self.evaluate else lt_probs[i]
                    thresholds[i] = cumulative
                if not self.evaluate:
                    survival = 1.0
                    total = 0.0
                    for i in range(low, high + 1):
                        estimates[i] = survival * max(0.0, thresholds[i])
                        total += estimates[i]
                        survival *= max(0.0, 1.0 - thresholds[i])
                    if abs(total - 1.0) >= 1e-8:
                        for i in range(len(estimates)):
                            estimates[i] /= total
        est_low = low
        est_high = high
        if self.censored_leadtime:
            est_low = self.max_leadtime
            est_high = self.max_leadtime
            for i in range(len(estimates)):
                estimates[i] = 0.0
            estimates[est_high] = 1.0
        periods: list[DemandPeriod] = []
        for i in range(size):
            dist = fit_aer(means[i], stds[i])
            estimate = dist
            mean = means[i]
            std = stds[i]
            if self.censored_demand:
                estimate = DiscreteDist.constant(int(math.ceil(self.max_demand)))
                mean = estimate.expectation()
                std = estimate.std()
            periods.append(DemandPeriod(groups[i], dist, estimate, mean, std))
        queue = FifoQueue()
        queue.reserve(self.pipeline_width + 1)
        queue.fill(self.pipeline_width)
        state = State(
            state_vector=queue,
            periods=periods,
            p=penalty,
            min_leadtime=low,
            max_leadtime=high,
            estimated_min_leadtime=est_low,
            estimated_max_leadtime=est_high,
            estimated_leadtime_probs=estimates,
            arrival_thresholds=thresholds,
            stochastic_leadtimes=stochastic,
            order_crossover=crossover,
            past_leadtimes=past_leadtimes,
            cycle_max_order_size=[],
            cycle_max_system_inv=[],
            arrival_draws=[],
        )
        for i in range(size):
            order, inventory = self.demand_over_leadtime_fractiles(state, i)
            order = min(order, self.max_order_size)
            state.cycle_max_order_size.append(order)
            state.cycle_max_system_inv.append(min(inventory, self.max_system_inv))
            state.max_order_size_limit = max(state.max_order_size_limit, order)
        if self.censored_demand:
            state.max_order_size_limit = self.max_order_size
        if crossover:
            state.arrival_draws = [0.0] * high
        state.max_order_size = state.cycle_max_order_size[0]
        state.max_system_inv = state.cycle_max_system_inv[0]
        state.order_constraint = min(state.max_order_size, state.max_system_inv)
        return state

    def demand_over_leadtime_fractiles(self, state: State, base_period: int) -> tuple[int, int]:
        # Incremental convolution and weighted mixtures avoid rebuilding every prefix.
        total = DiscreteDist.constant(0)
        on_leadtime = DiscreteDist.constant(0)
        over_leadtime = DiscreteDist.constant(0)
        weight = 0.0
        for leadtime in range(state.estimated_max_leadtime + 1):
            dist = state.periods[(base_period + leadtime) % len(state.periods)].estimate
            total = total.add(dist)
            prob = state.estimated_leadtime_probs[leadtime]
            if leadtime >= state.estimated_min_leadtime and prob > 0.0:
                if weight == 0.0:
                    on_leadtime = dist
                    over_leadtime = total
                else:
                    on_leadtime = on_leadtime.mix(dist, prob / (weight + prob))
                    over_leadtime = over_leadtime.mix(total, prob / (weight + prob))
                weight += prob
        ratio = state.p / (state.p + 1.0)
        return on_leadtime.fractile(ratio), over_leadtime.fractile(ratio)

    def modify_state_with_action(self, state: State, context: LostSalesContext, action: int) -> None:
        assert state.category == StateCategory.AWAIT_ACTION, "Expected an action state"
        assert 0 <= action and action < self.num_actions, "Order outside configured action domain"
        assert self.oracle_actions or action <= state.order_constraint, "Order exceeds action mask"
        state.state_vector.push_back(action)
        state.total_inv += action
        state.category = StateCategory.AWAIT_EVENT

    def modify_state_with_event(self, state: State, context: LostSalesContext) -> None:
        demand = state.periods[state.period].true_dist.sample(context.rng)
        # Nothing is allocated per period; crossover draws go to a state buffer. DynaPlex 1.14 does not count
        # an empty list towards garbage collection, so an empty list per period was never freed.
        draw = 0.0
        if state.stochastic_leadtimes:
            if state.order_crossover:
                # One draw per age that can arrive early, however many units it holds, so that every policy,
                # also one ordering beyond the action mask, draws the same random numbers.
                if state.min_leadtime == 0:
                    state.arrival_draws[0] = context.rng.random()
                for age in range(max(1, state.min_leadtime), state.max_leadtime):
                    state.arrival_draws[age] = context.rng.random()
            else:
                draw = context.rng.random()
        self.apply_event(state, context, demand, draw)

    def apply_event(self, state: State, context: LostSalesContext, demand: int, draw: float) -> None:
        """Apply an explicit demand/arrival event, enabling controlled replay.

        Without crossover, draw decides the arrivals; with crossover, state.arrival_draws has a draw per age.
        """
        assert state.category == StateCategory.AWAIT_EVENT, "Place an order before applying an event"
        assert demand >= 0, "Demand must be nonnegative"
        before_received = state.orders_received
        on_hand = state.state_vector.pop_front()
        arriving = 0
        if state.stochastic_leadtimes:
            if state.order_crossover:
                # Each unit in transit arrives independently, so orders may overtake each other.
                if state.min_leadtime == 0:
                    loc = len(state.state_vector) - 1
                    received = arriving_units(
                        state.state_vector[loc], state.arrival_thresholds[0], state.arrival_draws[0]
                    )
                    state.state_vector[loc] -= received
                    on_hand += received
                    if self.censored_leadtime:
                        state.past_leadtimes[0] += received
                        state.orders_received += received
                for age in range(max(1, state.min_leadtime), state.max_leadtime):
                    loc = self.pipeline_width - age
                    received = arriving_units(
                        state.state_vector[loc], state.arrival_thresholds[age], state.arrival_draws[age]
                    )
                    state.state_vector[loc] -= received
                    arriving += received
                    if self.censored_leadtime:
                        state.past_leadtimes[age] += received
                        state.orders_received += received
                loc = self.pipeline_width - state.max_leadtime
                arriving += state.state_vector[loc]
                if self.censored_leadtime:
                    state.past_leadtimes[state.max_leadtime] += state.state_vector[loc]
                    state.orders_received += state.state_vector[loc]
                state.state_vector[loc] = 0
            else:
                for age in range(state.min_leadtime, state.max_leadtime + 1):
                    loc = self.pipeline_width - max(1, age)
                    count = state.state_vector[loc]
                    if draw <= state.arrival_thresholds[age] and count > 0:
                        last_observed = age
                        if age == 0:
                            on_hand += count
                        else:
                            arriving += count
                        state.state_vector[loc] = 0
                        for older in range(age + 1, state.max_leadtime + 1):
                            loc = self.pipeline_width - older
                            if state.state_vector[loc] > 0:
                                last_observed = older
                                arriving += state.state_vector[loc]
                                state.state_vector[loc] = 0
                        if self.censored_leadtime:
                            for observed in range(age, last_observed + 1):
                                state.past_leadtimes[observed] += 1
                                state.orders_received += 1
                        break
        else:
            loc = self.pipeline_width - max(1, state.max_leadtime)
            if state.max_leadtime == 0:
                on_hand += state.state_vector[loc]
            else:
                arriving += state.state_vector[loc]
            state.state_vector[loc] = 0
        period = state.periods[state.period]
        if self.censored_demand and on_hand > 0:
            period.collect = True
        state.demand = demand
        sales = min(demand, on_hand)
        ending = on_hand - sales
        cost = float(ending) + state.p * (demand - sales)
        context.true_cost += cost
        context.total_demand += demand
        context.units_sold += sales
        context.lost_units += demand - sales
        context.cumulative_cost += float(ending) - state.p * sales if self.maximize_rewards else cost
        state.total_inv -= sales
        state.state_vector[0] = ending + arriving
        if self.censored_leadtime and state.orders_received > before_received:
            self.update_leadtime_statistics(state)
        if self.censored_demand and period.collect:
            self.update_demand_statistics(state, demand <= on_hand, sales)
        state.period = (state.period + 1) % len(state.periods)
        if self.censored_demand or self.censored_leadtime:
            order, inventory = self.demand_over_leadtime_fractiles(state, state.period)
            state.max_order_size = min(order, state.max_order_size_limit)
            state.max_system_inv = min(inventory, self.max_system_inv)
        else:
            state.max_order_size = state.cycle_max_order_size[state.period]
            state.max_system_inv = state.cycle_max_system_inv[state.period]
        state.order_constraint = max(0, min(state.max_system_inv - state.total_inv, state.max_order_size))
        context.time_elapsed += 1
        state.category = StateCategory.AWAIT_ACTION

    def update_leadtime_statistics(self, state: State) -> None:
        counts: list[int] = []
        for x in state.past_leadtimes:
            counts.append(x)
        received = state.orders_received
        if state.estimated_max_leadtime < self.max_leadtime:
            pending = 0
            for age in range(max(1, state.estimated_max_leadtime), self.max_leadtime):
                pending += state.state_vector[self.pipeline_width - age]
            if pending > 0:
                gap = self.max_leadtime - state.estimated_max_leadtime
                fractional = pending // gap
                distributed = 0
                for age in range(state.estimated_max_leadtime + 1, self.max_leadtime):
                    counts[age] += fractional
                    distributed += fractional
                counts[self.max_leadtime] += pending - distributed
                received += pending
        for i in range(self.max_leadtime + 1):
            state.estimated_leadtime_probs[i] = float(counts[i]) / received
        found = False
        total = 0.0
        for i in range(self.max_leadtime + 1):
            prob = state.estimated_leadtime_probs[i]
            total += prob
            if not found and prob > 0.0:
                state.estimated_min_leadtime = i
                found = True
            if abs(total - 1.0) < 1e-8:
                state.estimated_max_leadtime = i
                break

    def update_demand_statistics(self, state: State, uncensored: bool, observation: int) -> None:
        group = state.periods[state.period].group
        period = state.periods[group]
        period.count += 1
        old_size = len(period.past) - 1
        if observation > old_size:
            for _ in range(old_size + 1, observation):
                period.past.append(0)
                period.censored.append(0)
                period.cumulative.append(1)
            period.past.append(1)
            period.cumulative.append(0)
            period.censored.append(0 if uncensored else 1)
        else:
            period.past[observation] += 1
            old_size = observation - 1
            if not uncensored:
                period.censored[observation] += 1
        for i in range(old_size + 1):
            period.cumulative[i] += 1
        size = len(period.past)
        probs = np.zeros(size, dtype=np.float64)
        for i in range(size):
            probs[i] += float(period.past[i]) / period.count
            if period.censored[i] > 0 and i < size - 1:
                weight = float(period.censored[i]) / period.count
                probs[i] -= weight
                for j in range(i + 1, size):
                    probs[j] += period.past[j] * weight / period.cumulative[i]
        estimate = DiscreteDist.custom(probs, 0)
        for i in range(len(state.periods)):
            other = state.periods[i]
            if other.group == group:
                other.estimate = estimate
                other.mean = estimate.expectation()
                other.std = estimate.std()
                if i != group:
                    other.count = period.count
                    other.past = []
                    for x in period.past:
                        other.past.append(x)
                    other.cumulative = []
                    for x in period.cumulative:
                        other.cumulative.append(x)
                    other.censored = []
                    for x in period.censored:
                        other.censored.append(x)

    def write_action_validity(self, state: State, valid: Validity) -> None:
        for action in range(self.num_actions):
            valid.set(action, self.oracle_actions or action <= state.order_constraint)


@featurizer
@dataclass(slots=True)
class ZeroShotFeaturizer(Featurizer):
    mdp: Final[ZeroShotLostSalesMDP]
    v: Final[GlobalStateWriter]

    def write_features(self, state: State) -> None:
        if self.mdp.train_stochastic_leadtimes:
            self.v.append(1.0 if state.order_crossover else 0.0)
        self.v.append(state.p)
        for i in range(len(state.state_vector)):
            self.v.append(float(state.state_vector[i]))
        if self.mdp.train_stochastic_leadtimes:
            for i in range(self.mdp.max_leadtime, -1, -1):
                self.v.append(state.estimated_leadtime_probs[i])
        else:
            self.v.append(float(state.min_leadtime))
        if self.mdp.train_cyclic_demand:
            self.v.append(float(len(state.periods)))
            for i in range(self.mdp.max_num_cycles):
                period = state.periods[(state.period + i) % len(state.periods)]
                self.v.append(period.mean)
                self.v.append(period.std)
        else:
            self.v.append(state.periods[0].mean)
            self.v.append(state.periods[0].std)


# The paper's super-MDP dimensions; every case instance fits within them.
DIMENSIONS = {
    "max_p": 100.0,
    "max_demand": 12.0,
    "max_leadtime": 10,
    "max_num_cycles": 7,
    "train_stochastic_leadtimes": True,
    "train_cyclic_demand": True,
}


def instance_config(instance, mode: str, dimensions: dict) -> dict:
    """The model of a case instance: "known" gives true parameters; "demand", "leadtime" or "both" are estimated."""
    if mode not in instance.modes:
        raise ValueError(f"{instance.id} is not evaluated with mode {mode!r}")
    config = dimensions | instance.parameters
    config.update(
        evaluate=True,
        censored_demand=mode in ("demand", "both"),
        censored_leadtime=mode in ("leadtime", "both"),
        # With censored demand the legacy objective is holding cost minus sales revenue.
        maximize_rewards=mode in ("demand", "both"),
    )
    if "leadtime_distribution" in config:
        probs = list(config["leadtime_distribution"])
        length = config["max_leadtime"] + 1
        if any(probs[length:]):
            raise ValueError(f"{instance.id} needs a larger max_leadtime")
        config["leadtime_distribution"] = (probs + [0.0] * length)[:length]
    return config
