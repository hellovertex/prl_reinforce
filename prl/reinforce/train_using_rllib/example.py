from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.multi_agent.utils import make_multi_agent_env

from prl.reinforce.train_using_rllib.runner import run_rainbow_vs_baseline_example

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
if __name__ == '__main__':
    env_cfg = {'env_wrapper_cls': AugmentObservationWrapper,
               'agents': {0: BASELINE_AGENT,
                          1: TRAINABLE_AGENT},
               'n_players': 2,
               'starting_stack_size': 1000,
               'blinds': [25, 50],
               'num_envs': 2,
               'mask_legal_moves': True
               }
    env_cls = make_multi_agent_env(env_cfg)
    # dummy_ctx = EnvContext(env_config={},
    #                        worker_index=0,  # 0 for local worker, >0 for remote workers.
    #                        vector_index=0,  # uniquely identify env when there are multiple envs per worker
    #                        remote=False,  # individual sub-envvs should be @ray.remote actors
    #                        num_workers=0,  # 0 for only local
    #                        recreated_worker=False
    #                        )
    # env = env_cls(dummy_ctx)
    # print(env)
    run_rainbow_vs_baseline_example(env_cls)
