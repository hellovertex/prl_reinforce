from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.utils.typing import AlgorithmConfigDict


def make_rainbow_config(config: AlgorithmConfigDict):
    config["n_step"] = 1  # todo: make configurable
    config["noisy"] = True
    config["num_atoms"] = 51
    config["v_min"] = -5.0
    config["v_max"] = 5.0
    return config


def get_distributed_rainbow(rainbow_config_dict):
    return ApexDQN(rainbow_config_dict)
