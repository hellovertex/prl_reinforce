import os
from typing import Type

import gin
import ray.rllib.algorithms.registry
from prl.baselines.agents.policies import StakeLevelImitationPolicy
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.multi_agent.utils import make_multi_agent_env
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.algorithms.simple_q import SimpleQ, SimpleQConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from prl.reinforce.agents.our_models import TrainableModelType, CustomTorchModel
from prl.reinforce.agents.rainbow import make_rainbow_config
from prl.reinforce.train_using_rllib.prl_callbacks.our_callbacks import PRLToRllibCallbacks

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
    # if agent_id in [0, 2, 3, 5]:
    if agent_id == 0:
        return BASELINE_POLICY
    else:
        return RAINBOW_POLICY


@gin.configurable
def run(algo_class=ApexDQN,
        prl_baseline_model_ckpt_path="",
        min_sample_timesteps_per_iteration=100,
        num_steps_sampled_before_learning_starts=1000,
        max_episodes=100,
        replay_buffer_capacity=5000,
        max_iter_per_episode=10,
        ckpt_interval=10,
        algo_ckpt_dir="./algo_ckpt"):
    env_config = {'env_wrapper_cls': AugmentObservationWrapper,
                  'agents': {0: BASELINE_POLICY,
                             1: RAINBOW_POLICY,
                             # 2: BASELINE_POLICY,
                             # 3: BASELINE_POLICY,
                             # 4: RAINBOW_POLICY,
                             # 5: BASELINE_POLICY
                             },
                  'n_players': 2,
                  'starting_stack_size': 1000,
                  'blinds': [25, 50],
                  # 'num_envs': 2,
                  'mask_legal_moves': True}
    env_cls: Type[MultiAgentEnv] = make_multi_agent_env(env_config)
    policies = {RAINBOW_POLICY: PolicySpec(  # no policy_class specified --> will fall back to return algo_class
                    # config={
                    #     "model": {**MODEL_DEFAULTS,
                    #               "fcnet_hiddens": [512, 512],
                    #               "_disable_preprocessor_api": True,
                    #               # "_disable_preprocessor_api": True,
                    #               # "_disable_action_flattening": True,
                    #               "custom_model": CustomTorchModel,  # todo how to set custom NN ?
                    #               "custom_model_config": {}
                    #               },
                    #     "framework": "torch",
                    #     # "observation_space": observation_space
                    # }
                ),
                BASELINE_POLICY: PolicySpec(policy_class=StakeLevelImitationPolicy,
                                            config={'path_to_torch_model_state_dict': prl_baseline_model_ckpt_path}),
            }
    # observation_space = env_cls(None).observation_space['obs']
    # conf = ApexDQNConfig()
    conf = ApexDQNConfig()
    conf = conf.environment(env=env_cls)
    conf = conf.training(gamma=0.9,
                         replay_buffer_config={**algo_class.get_default_config()["replay_buffer_config"],
                                               "capacity": replay_buffer_capacity},

                         )
    conf = conf.resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    conf = conf.rollouts(num_rollout_workers=3,
                         num_envs_per_worker=1,
                         rollout_fragment_length=50,
                         horizon=max_iter_per_episode,
                         )
    conf = conf.evaluation(evaluation_interval=1)
    conf = conf.debugging(log_level="DEBUG")
    conf = conf.reporting(min_sample_timesteps_per_iteration=min_sample_timesteps_per_iteration,
                          )
    conf = conf.callbacks(PRLToRllibCallbacks)
    conf = conf.multi_agent(policies=policies,
                            policy_mapping_fn=policy_selector,
                            policies_to_train=[RAINBOW_POLICY])
    conf = conf.framework(framework="torch")
    conf = conf.experimental(_disable_preprocessor_api=True)

    algo_config = make_rainbow_config(conf)  # APEXDQN is now distributed Rainbow
    # algo_config = conf

    results = TrainRunner().run(algo_class,
                                algo_config,
                                algo_ckpt_dir,
                                ckpt_interval)


if __name__ == '__main__':
    gin.parse_config_file('./gin_configs/config_example.gin')
    run()
