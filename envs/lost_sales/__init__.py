"""Lost-sales inventory control: three environments, the paper's cases, and their evaluation.

    lost_sales                       one instance with known parameters; an agent per instance
    zero_shot_lost_sales_stationary  a super-MDP of stationary instances; the workshop environment
    zero_shot_lost_sales_large       the paper's super-MDP, with cyclic demand and stochastic lead times

cases/ holds the instances every environment is evaluated on, and evaluate.py the protocol.
"""
