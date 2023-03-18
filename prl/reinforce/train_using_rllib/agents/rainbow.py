from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.utils.typing import AlgorithmConfigDict


def make_rainbow_config(config: ApexDQNConfig):
    config.n_step = 1
    config.noisy = True
    config.num_atoms = 51
    config.v_max = 5.0
    config.v_min = -5.0
    config.hiddens = [512, 512]
    return config
