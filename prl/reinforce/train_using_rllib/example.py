import os
from typing import Type

import gin
from prl.baselines.agents.policies import StakeLevelImitationPolicy
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.multi_agent.utils import make_multi_agent_env
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from prl.reinforce.agents.our_models import TrainableModelType
from prl.reinforce.agents.rainbow import make_rainbow_config
from prl.reinforce.train_using_rllib.prl_callbacks.our_callbacks import OurRllibCallbacks

RAINBOW_POLICY = "ApexDQN"
BASELINE_POLICY = "StakeImitation"

from prl.reinforce.train_using_rllib.runner import TrainRunner


# ray.tune.run(ApexTrainer,
#              # config=config,  # todo check whether bottom config overwrites ApexDqnConfig
#              config={
#                  "env": "CartPole-v0",  # todo check how to set our env
#                  "num_gpus": 0,
#                  "num_workers": 1,
#                  "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#              },
#              )


def policy_selector(agent_id, episode, **kwargs):
    # if "player" not in agent_id:
    #     raise ValueError("WRONG AGENT ID")
    if agent_id == 0:
        return RAINBOW_POLICY
    else:
        return BASELINE_POLICY


@gin.configurable
def run(algo_class=ApexDQN,
        prl_baseline_model_ckpt_path="",
        min_sample_timesteps_per_iteration=10,
        num_steps_sampled_before_learning_starts=10,
        replay_buffer_capacity=5000,
        max_episodes=100,
        max_iter_per_episode=10,
        ckpt_interval=10,
        algo_ckpt_dir="./algo_ckpt"):
    env_config = {'env_wrapper_cls': AugmentObservationWrapper,
                  'agents': {0: "Baseline",
                             1: "Trainable"},
                  'n_players': 2,
                  'starting_stack_size': 1000,
                  'blinds': [25, 50],
                  'num_envs': 2,
                  'mask_legal_moves': True
                  }
    env_cls: Type[MultiAgentEnv] = make_multi_agent_env(env_config)
    observation_space = env_cls(None).observation_space['obs']
    algo_config = {
        "env": env_cls,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 8,
        # "num_cpus_per_worker": 8,
        "max_episodes": max_episodes,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 50,
        "evaluation_interval": 1,
        # "train_batch_size": 50,
        # "metrics_num_episodes_for_smoothing": 20,
        "log_level": "WARN",
        "min_sample_timesteps_per_iteration": min_sample_timesteps_per_iteration,
        "num_steps_sampled_before_learning_starts": num_steps_sampled_before_learning_starts,
        "horizon": max_iter_per_episode,
        "callbacks": OurRllibCallbacks,
        "replay_buffer_config": {**algo_class.get_default_config()["replay_buffer_config"],
                                 "capacity": replay_buffer_capacity},
        '_disable_preprocessor_api': True,
        "multiagent": {
            "policies_to_train": [RAINBOW_POLICY],
            "policies": {
                RAINBOW_POLICY: PolicySpec(
                    config={
                        "model": {**MODEL_DEFAULTS,
                                  "custom_model": 0,  # todo how to set custom NN ?
                                  "custom_model_config": {}},
                        "framework": "torch",
                        "observation_space": observation_space
                    }
                ),
                BASELINE_POLICY: PolicySpec(policy_class=StakeLevelImitationPolicy,
                                            config={'path_to_torch_model_state_dict': prl_baseline_model_ckpt_path}),
            },
            "policy_mapping_fn": policy_selector,
        },
        "framework": "torch",
    }
    algo_config = make_rainbow_config(algo_config)  # APEXDQN is now distributed Rainbow

    results = TrainRunner().run(algo_class,
                                algo_config,
                                algo_ckpt_dir,
                                ckpt_interval)


if __name__ == '__main__':
    gin.parse_config_file('./gin_configs/config_example.gin')
    run()
