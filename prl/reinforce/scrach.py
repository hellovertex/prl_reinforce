# todo:
# 1. run self play with rainbow agent that has seer mode enabled POSTFLOP only
# 2. this will mean it learns to play perfect ranges
# 3. rl training vs random agent
from dataclasses import dataclass, field
from typing import List
import os
import os
import pprint
import time

import numpy as np
import torch
from prl.baselines.agents.tianshou_agents import TianshouRandomAgent
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import RainbowPolicy, MultiAgentPolicyManager
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from prl.baselines.agents.tianshou_policies import get_rainbow_config
from prl.baselines.examples.examples_tianshou_env import make_vectorized_prl_env, make_vectorized_pettingzoo_env

import click
from hydra.core.config_store import ConfigStore
# todo: implement RL trainer vs Random agent
#  what do we need in terms of experimental evaluation (tables, mbb charts)
#  what do we need in terms of reproducability?
from omegaconf import DictConfig, OmegaConf
import hydra


class Reward:
    def __init__(self):
        self.reward = 0


@dataclass
class _TrainConfig:
    num_players: int
    device: str
    buffer_sizes: List[int]
    target_update_freqs: List[int]
    dir_suffix: str
    obs_stack: int
    alpha: float
    beta: float
    beta_final: int
    beta_anneal_step: int
    weight_norm: bool
    epoch: int
    step_per_epoch: int
    step_per_collect: int
    episode_per_test: int
    batch_size: int
    update_per_step: float
    eps_train: float
    eps_train_final: float
    eps_test: float
    no_priority: bool
    load_ckpt: bool


# @dataclass
# class TrainConfig:
#     debug: str
#     baselines: List[_TrainConfig] = field(default_factory=_TrainConfig)


class TrainEval:
    def __init__(self, config: _TrainConfig):
        self.config = config

    def _run(self, buffer_size, target_update_freq):
        dir_suffix = f"num_players={target_update_freq},buffer_size={buffer_size}"
        win_rate_early_stopping = np.inf,
        best_rew = -np.inf
        learning_agent_ids = [0]
        logdir = [".", "v3", "rainbow_self_play", dir_suffix]
        ckpt_save_path = os.path.join(
            *logdir, f'ckpt.pt'
        )
        # environment config
        starting_stack = 20000
        stack_sizes = [starting_stack for _ in range(self.config.num_players)]
        agents = [f'p{i}' for i in range(self.config.num_players)]
        sb = 50
        bb = 100
        env_config = {"env_wrapper_cls": AugmentObservationWrapper,
                      # "stack_sizes": [100, 125, 150, 175, 200, 250],
                      "stack_sizes": stack_sizes,
                      "multiply_by": 1,  # use 100 for floats to remove decimals but we have int stacks
                      "scale_rewards": False,  # we do this ourselves
                      "blinds": [sb, bb]}
        # env = init_wrapped_env(**env_config)
        # obs0 = env.reset(config=None)
        num_envs = 31
        mc_model_ckpt_path = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/ckpt/ckpt.pt"
        venv, wrapped_env = make_vectorized_pettingzoo_env(num_envs=num_envs,
                                                           single_env_config=env_config,
                                                           agent_names=agents,
                                                           mc_model_ckpt_path=mc_model_ckpt_path)

        params = {'device': self.config.device,
                  'load_from_ckpt': ckpt_save_path,
                  'lr': 1e-6,
                  'num_atoms': 51,
                  'noisy_std': 0.1,
                  'v_min': -6,
                  'v_max': 6,
                  'estimation_step': 3,
                  'target_update_freq': target_update_freq  # training steps
                  }
        rainbow_config = get_rainbow_config(params)
        rainbow_policy = RainbowPolicy(**rainbow_config)
        if self.config.load_ckpt:
            try:
                rainbow_policy.load_state_dict(torch.load(ckpt_save_path, map_location=self.config.device))
            except FileNotFoundError:
                # initial training, no ckpt created yet, ignore silently
                pass
        # # 'load_from_ckpt_dir': None
        rainbow = RainbowPolicy(**rainbow_config)
        random_agent = TianshouRandomAgent()
        policy = MultiAgentPolicyManager([
            rainbow,
            random_agent,
            random_agent,
            random_agent,
            random_agent,
            random_agent
        ], wrapped_env)  # policy is made from PettingZooEnv
        # policy = RainbowPolicy(**rainbow_config)

        buffer = PrioritizedVectorReplayBuffer(
            total_size=buffer_size,
            buffer_num=len(venv),
            ignore_obs_next=False,  # enable for framestacking
            save_only_last_obs=False,  # enable for framestacking
            stack_num=self.config.obs_stack,
            alpha=self.config.alpha,
            beta=self.config.beta,
            weight_norm=self.config.weight_norm
        )
        train_collector = Collector(policy, venv, buffer, exploration_noise=True)
        test_collector = Collector(policy, venv, exploration_noise=True)

        def train_fn(epoch, env_step, beta=self.config.beta):
            # linear decay in the first 10M steps
            if env_step <= 1e7:
                eps = self.config.eps_train - env_step / 1e7 * \
                      (self.config.eps_train - self.config.eps_train_final)
            else:
                eps = self.config.eps_train_final
            for aid in learning_agent_ids:
                policy.policies[agents[aid]].set_eps(eps)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/eps": eps})
            if not self.config.no_priority:
                if env_step <= self.config.beta_anneal_step:
                    beta = beta - env_step / self.config.beta_anneal_step * \
                           (beta - self.config.beta_final)
                else:
                    beta = self.config.beta_final
                buffer.set_beta(beta)
                if env_step % 1000 == 0:
                    logger.write("train/env_step", env_step, {"train/beta": beta})

        def test_fn(epoch, env_step):
            for aid in learning_agent_ids:
                policy.policies[agents[aid]].set_eps(self.config.eps_test)

        def save_best_fn(policy):
            for aid in learning_agent_ids:
                model_save_path = os.path.join(
                    *logdir, f'policy_{aid}.pth'
                )
                torch.save(
                    policy.policies[agents[aid]].state_dict(), model_save_path
                )

        def stop_fn(mean_rewards):
            return mean_rewards >= win_rate_early_stopping

        max_reward = Reward()

        def reward_metric(rews):
            # The reward at index 0 is the reward relative to observer
            rew = np.mean(rews[:, learning_agent_ids[0]])
            if rew > max_reward.reward:
                max_reward.reward = rew
            return rew

        log_path = os.path.join(*logdir)
        writer = SummaryWriter(log_path)
        # writer.add_text("args", str(args))
        logger = TensorboardLogger(writer)

        def save_checkpoint_fn(epoch: int,
                               env_step: int,
                               gradient_step: int) -> str:
            # for aid in learning_agent_ids:
            # assume learning agent is at index 0
            torch.save({
                'epoch': epoch,
                'net': policy.state_dict(),
                'model': rainbow_config['model'].state_dict(),
                'env_step': env_step,
                'optim': rainbow_config['optim'].state_dict(),
            }, ckpt_save_path)
            return ckpt_save_path

        # test train_collector and start filling replay buffer
        train_collector.collect(n_step=self.config.batch_size * num_envs)
        trainer = OffpolicyTrainer(policy=policy,
                                   train_collector=train_collector,
                                   test_collector=test_collector,
                                   max_epoch=self.config.epoch,  # set stop_fn for early stopping
                                   step_per_epoch=self.config.step_per_epoch,  # num transitions per epoch
                                   step_per_collect=self.config.step_per_collect,
                                   # step_per_collect -> update network -> repeat
                                   episode_per_test=self.config.episode_per_test,
                                   # games to play for one policy evaluation
                                   batch_size=self.config.batch_size,
                                   update_per_step=self.config.update_per_step,  # fraction of steps_per_collect
                                   train_fn=train_fn,
                                   test_fn=test_fn,
                                   stop_fn=None,  # early stopping
                                   save_best_fn=save_best_fn,
                                   save_checkpoint_fn=save_checkpoint_fn,
                                   resume_from_log=self.config.load_ckpt,
                                   reward_metric=reward_metric,
                                   logger=logger,
                                   verbose=True,
                                   show_progress=True,
                                   test_in_train=False  # whether to test in training phase
                                   )
        result = trainer.run()
        t0 = time.time()
        pprint.pprint(result)
        print(f'took {time.time() - t0} seconds')
        # pprint.pprint(result)
        # watch()
        return max_reward.reward

    def run(self):
        for buffer_size in self.config.buffer_sizes:
            for target_update_freq in self.config.target_update_freqs:
                self._run(buffer_size, target_update_freq)


# cs = ConfigStore.instance()
# # Registering the Config class with the name 'config'.
# cs.store(name="config", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf/training", config_name='config')
def main(cfg):
    """Run with `python scratch.py`, no need to overwrite in config group"""
    """
    for b in cfg.baselines:
    print(type(b))    
    <class 'str'>
    <class 'str'>
    """
    conf = _TrainConfig(**cfg.baselines.random_agent)
    TrainEval(conf).run()


if __name__ == "__main__":
    main()
