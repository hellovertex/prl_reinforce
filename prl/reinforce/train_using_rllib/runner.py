import os

from prl.baselines.agents.policies import StakeLevelImitationPolicy
from ray.rllib.algorithms.simple_q import SimpleQ
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import PolicySpec

from prl.reinforce.agents.our_models import TrainableModelType
from prl.reinforce.train_using_rllib.our_callbacks import OurRllibCallbacks
from prl.environment.multi_agent.utils import make_multi_agent_env
RAINBOW_POLICY = "SimpleQ"
BASELINE_POLICY = "StakeImitation"
DistributedRainbow = SimpleQ


def policy_selector(agent_id, episode, **kwargs):
    # if "player" not in agent_id:
    #     raise ValueError("WRONG AGENT ID")
    if agent_id == 0:
        return RAINBOW_POLICY
    else:
        return BASELINE_POLICY


def run_rainbow_vs_baseline_example(env_cls):
    """Run heuristic policies vs a learned agent.
    under construction.
    """
    observation_space = env_cls(None).observation_space['obs']
    config = {
        "env": env_cls,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "num_envs_per_worker": 4,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        "metrics_num_episodes_for_smoothing": 20,
        "log_level": "DEBUG",
        "horizon": 100,
        "callbacks": OurRllibCallbacks,
        "replay_buffer_config": {**SimpleQ.get_default_config()["replay_buffer_config"],
                                 "capacity": 1000},
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
                                            config={'path_to_torch_model_state_dict': 'ckpt.pt'}),
            },
            "policy_mapping_fn": policy_selector,
        },
        "framework": "torch",
    }
    # todo uncomment this, when we are ready to use real Rainbow
    # config = make_rainbow_config(config)
    # rainbow_agent = get_distributed_rainbow(config)
    algo = DistributedRainbow(config=config)

    for _ in range(3):
        results = algo.train()
        # Timesteps reached.
        if "policy_always_same_reward" not in results["hist_stats"]:
            reward_diff = 0
            continue
        # reward_diff = sum(results["hist_stats"]["policy_learned_reward"])
        # if results["timesteps_total"] > 100000:
        #     break
        # # Reward (difference) reached -> all good, return.
        # elif reward_diff > 1000:
        #     return
