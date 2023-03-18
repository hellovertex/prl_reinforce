import os
from typing import Type

import gin
import hydra
import ray.rllib.algorithms.registry
from prl.baselines.agents.rllib_policies import AlwaysCallingPolicy
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.vectorizer import AgentObservationType
from prl.environment.multi_agent.utils import make_multi_agent_env
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms.apex_dqn import ApexDQN, ApexDQNConfig
from ray.rllib.algorithms.simple_q import SimpleQ, SimpleQConfig
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from prl.reinforce.train_using_rllib.agents.our_models import TrainableModelType, CustomTorchModel
from prl.reinforce.train_using_rllib.agents.rainbow import make_rainbow_config
from prl.reinforce.train_using_rllib.prl_callbacks.our_callbacks import PRLToRllibCallbacks

RAINBOW_POLICY = "ApexDQN"
BASELINE_POLICY = "AlwaysCallingPolicy"

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


def run(algo_class=ApexDQN,
        prl_baseline_model_ckpt_path="",
        min_sample_timesteps_per_iteration=100,
        num_steps_sampled_before_learning_starts=1000,
        max_episodes=100,
        replay_buffer_capacity=5000,
        max_iter_per_episode=10,
        ckpt_interval=10):
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
                  'mask_legal_moves': True,
                  'agent_observation_mode': AgentObservationType.SEER}
    env_cls: Type[MultiAgentEnv] = make_multi_agent_env(env_config)
    policies = {RAINBOW_POLICY: PolicySpec(),  # empty defaults to agent_class policy --> RAINBOW
                BASELINE_POLICY: PolicySpec(policy_class=AlwaysCallingPolicy,
                                            config={}),
                }
    # observation_space = env_cls(None).observation_space['obs']
    replay_buffer_config = {**algo_class.get_default_config()["replay_buffer_config"],
                            "capacity": replay_buffer_capacity}
    conf = ApexDQNConfig()
    conf = conf.environment(env=env_cls)
    conf = conf.training(gamma=0.9,
                         replay_buffer_config=algo_class.get_default_config(),  # replay_buffer_config,
                         )
    conf = conf.resources(num_gpus=1, )
    conf = conf.rollouts(num_rollout_workers=os.cpu_count(),
                         # # use num_envs_per_woker > 1 with remote_worker_envs
                         # # to spawn envs in new processes. causes significant
                         # # overhead but may be worth if stepping the env is slow
                         # num_envs_per_worker=4,
                         # remote_worker_envs=True,
                         # rollout_fragment_length= 50,
                         horizon=max_iter_per_episode,
                         )
    conf = conf.evaluation(evaluation_interval=10)
    conf = conf.debugging(log_level="INFO", log_sys_usage=True)
    # conf = conf.reporting(min_sample_timesteps_per_iteration=min_sample_timesteps_per_iteration,
    #                       )
    conf = conf.callbacks(PRLToRllibCallbacks)
    conf = conf.multi_agent(policies=policies,
                            policy_mapping_fn=policy_selector,
                            policies_to_train=[RAINBOW_POLICY])
    conf = conf.framework(framework="torch")
    conf = conf.experimental(_disable_preprocessor_api=True)  # important! this prevents flattening obs_dict
    # apexdqn
    conf.replay_buffer_config = replay_buffer_config
    algo_config = make_rainbow_config(conf)  # APEXDQN is now distributed Rainbow
    # algo_config = conf

    results = TrainRunner().run(algo_class,
                                algo_config,
                                './tmp',
                                ckpt_interval)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(cfg):
    # conf = TrainConfig(**cfg.training)
    run()


if __name__ == '__main__':
    main()
