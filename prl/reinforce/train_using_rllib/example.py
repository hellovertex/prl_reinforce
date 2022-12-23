from prl.environment.Wrappers.augment import AugmentObservationWrapper
import os
from typing import Type

import gin
from prl.baselines.agents.policies import StakeLevelImitationPolicy
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.simple_q import SimpleQ
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from prl.reinforce.agents.our_models import TrainableModelType
from prl.reinforce.train_using_rllib.our_callbacks import OurRllibCallbacks
from prl.environment.multi_agent.utils import make_multi_agent_env
import os
from typing import Type

import gin
from prl.baselines.agents.policies import StakeLevelImitationPolicy
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.multi_agent.utils import make_multi_agent_env
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.simple_q import SimpleQ
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from prl.reinforce.agents.our_models import TrainableModelType
from prl.reinforce.train_using_rllib.our_callbacks import OurRllibCallbacks

RAINBOW_POLICY = "SimpleQ"
BASELINE_POLICY = "StakeImitation"
DistributedRainbow = SimpleQ
from prl.reinforce.train_using_rllib.runner import TrainRunner

BASELINE_AGENT = "Baseline"
TRAINABLE_AGENT = "Trainable"


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
def run(prl_baseline_model_ckpt_path="",
        min_sample_timesteps_per_iteration=10,
        num_steps_sampled_before_learning_starts=10,
        max_episodes=100,
        max_iter_per_episode=10,
        ckpt_interval=10,
        algo_ckpt_dir="./algo_ckpt"):
    env_config = {'env_wrapper_cls': AugmentObservationWrapper,
                  'agents': {0: BASELINE_AGENT,
                             1: TRAINABLE_AGENT},
                  'n_players': 2,
                  'starting_stack_size': 1000,
                  'blinds': [25, 50],
                  'num_envs': 2,
                  'mask_legal_moves': True
                  }
    env_cls: Type[MultiAgentEnv] = make_multi_agent_env(env_config)
    observation_space = env_cls(None).observation_space['obs']
    algo_class = DistributedRainbow
    algo_config = {
        "env": env_cls,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 10,
        "train_batch_size": 10,
        "metrics_num_episodes_for_smoothing": 20,
        "log_level": "DEBUG",
        "min_sample_timesteps_per_iteration": min_sample_timesteps_per_iteration,
        "num_steps_sampled_before_learning_starts": num_steps_sampled_before_learning_starts,
        "horizon": max_iter_per_episode,
        "callbacks": OurRllibCallbacks,
        "replay_buffer_config": {**SimpleQ.get_default_config()["replay_buffer_config"],
                                 "capacity": 100},
        '_disable_preprocessor_api': True,
        "multiagent": {
            "policies_to_train": [RAINBOW_POLICY],
            "policies": {
                RAINBOW_POLICY: PolicySpec(
                    config={
                        "model": {**MODEL_DEFAULTS,
                                  "custom_model": TrainableModelType.CUSTOM_TORCH_MLP.name,
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
    # todo uncomment this, when we are ready to use real Rainbow
    # config = make_rainbow_config(config)
    # rainbow_agent = get_distributed_rainbow(config)

    results = TrainRunner().run(algo_class,
                                algo_config,
                                algo_ckpt_dir,
                                ckpt_interval)


if __name__ == '__main__':
    gin.parse_config_file('./gin_configs/config_example.gin')
    run()
