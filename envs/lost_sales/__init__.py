"""Lost-sales inventory control: three environments, the paper's cases, and their evaluation.

    zero_shot_stationary  a super-MDP of stationary instances; the workshop environment
    zero_shot_large       the paper's super-MDP, with cyclic demand and stochastic lead times
    single_instance       one instance with known parameters; an agent per instance

cases/ holds the instances every environment is evaluated on, evaluate.py the protocol, and distributions.py
the demand and arrival distributions the models share.
"""
