import numpy as np
import pytest
from prl.baselines.examples.examples_tianshou_env import make_vectorized_pettingzoo_env
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.vectorizer import AgentObservationType

from prl.reinforce.train_eval import RegisteredAgent


# @pytest.fixture
def test_pettingzoo_env_step():
    num_envs = 1
    agent_names = [RegisteredAgent.always_all_in.__name__,
                   RegisteredAgent.always_fold.__name__]
    env_config = {}
    num_players = len(agent_names)
    env_config["env_wrapper_cls"] = AugmentObservationWrapper
    env_config["stack_sizes"] = [20000 for _ in
                                 range(num_players)]
    env_config["multiply_by"] = 1
    env_config["agent_observation_mode"] = AgentObservationType.SEER
    env_config['scale_rewards'] = False
    env_config['blinds'] = [50, 100]
    venv, wrapped_env = make_vectorized_pettingzoo_env(
        num_envs=num_envs,
        single_env_config=env_config,
        agent_names=agent_names,
        mc_model_ckpt_path="",
        debug_reset_config_state_dict=None)
    action = np.zeros(1)
    obs = wrapped_env.reset()
    obs = wrapped_env.step(0)
    assert True
