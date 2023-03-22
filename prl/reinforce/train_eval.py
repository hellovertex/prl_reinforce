# todo:
# 1. run self play with rainbow agent that has seer mode enabled POSTFLOP only
# 2. this will mean it learns to play perfect ranges
# 3. rl training vs random agent
import os
import pprint
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gym
import hydra
import numpy as np
import torch
from prl.baselines.agents.tianshou_agents import TianshouRandomAgent, \
    TianshouAlwaysFoldAgentDummy, \
    TianshouCallingStation, OracleAgent, TianshouALLInAgent
from prl.baselines.agents.tianshou_policies import get_rainbow_config
from prl.baselines.examples.examples_tianshou_env import make_vectorized_pettingzoo_env
from prl.environment.Wrappers.vectorizer import AgentObservationType
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import RainbowPolicy, MultiAgentPolicyManager, PPOPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from prl.reinforce.agents.rainbow import Rainbow


class Reward:
    def __init__(self):
        self.best_reward = -np.inf
        self.cumulated = 0


@dataclass
class RLConfig:
    device: str
    buffer_sizes: List[int]
    target_update_freqs: List[int]
    eps_decay_in_first_n_steps: int
    n_step_lookahead: int
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


@dataclass
class EnvConfig:
    starting_stack_size: int
    blinds: List[int]
    scale_rewards: bool
    agent_observation_mode: AgentObservationType  # CARD_KNOWLEDGE or SEER


@dataclass
class RainbowConfig:
    device: str
    load_from_ckpt: bool
    lr: float
    num_atoms: int
    noisy_std: float
    v_min: int
    v_max: int
    estimation_step: int
    target_update_freq: int


@dataclass
class OracleAgentConfig:
    model_ckpt_path: str
    model_hidden_dims: Tuple[int]
    device: str
    num_players: int
    observation_space: Optional[gym.Space] = None
    action_space: Optional[gym.Space] = None
    flatten_input: bool = False


class RegisteredAgent:
    # todo make rainbow learner configurable
    rainbow_learner = RainbowPolicy
    oracle = OracleAgent
    random_agent = TianshouRandomAgent
    always_all_in = TianshouALLInAgent
    calling_station = TianshouCallingStation
    always_fold = TianshouAlwaysFoldAgentDummy

    # register your agents here

    @staticmethod
    def build_agent_instance(name: str,
                             rainbow_config: Optional[RainbowConfig] = None,
                             oracle_config: Optional[OracleAgentConfig] = None,
                             ):
        """Monkey-patched agent creation function that is called inside train_eval.run() to make RL agent players"""
        if name.lower() == 'rainbow_learner':
            assert rainbow_config, "Need config to make rainbow_learner"
            params = {'device': rainbow_config.device,
                      'load_from_ckpt': rainbow_config.load_from_ckpt,
                      'lr': rainbow_config.lr,
                      'num_atoms': rainbow_config.num_atoms,
                      'noisy_std': rainbow_config.noisy_std,
                      'v_min': rainbow_config.v_min,
                      'v_max': rainbow_config.v_max,
                      'estimation_step': rainbow_config.estimation_step,
                      'target_update_freq': rainbow_config.target_update_freq
                      # training steps
                      }
            rainbow_config = get_rainbow_config(params)
            rainbow = Rainbow(**rainbow_config)
            return rainbow, rainbow_config
        elif name.lower() == 'oracle':
            return OracleAgent(
                model_ckpt_path=oracle_config.model_ckpt_path,
                flatten_input=oracle_config.flatten_input,
                model_hidden_dims=oracle_config.model_hidden_dims,
                device=oracle_config.device,
                observation_space=oracle_config.observation_space,
                num_players=oracle_config.num_players,
                action_space=oracle_config.action_space
            ), {}
        elif name.lower() == 'random_agent':
            return TianshouRandomAgent(), {}
        elif name.lower() == 'calling_station':
            return TianshouCallingStation(), {}
        elif name.lower() == 'always_fold':
            return TianshouAlwaysFoldAgentDummy(), {}
        elif name.lower() == 'always_all_in':
            return TianshouALLInAgent(), {}
        else:
            raise NotImplementedError("Agent not registered in class: RegisterdAgent")


@dataclass
class TrainConfig:
    experiment_name: str
    num_parallel_envs: int
    logdir_subdirs: List[str]
    rl_config: RLConfig
    env_config: EnvConfig
    agents: List[RegisteredAgent]
    learning_agent_ids: List[int]
    oracle_ckpt_dir: Optional[str] = None


# @dataclass
# class TrainConfig:
#     debug: str
#     baselines: List[_TrainConfig] = field(default_factory=_TrainConfig)

from prl.environment.Wrappers.augment import AugmentObservationWrapper


class TrainEval:
    def __init__(self,
                 config: TrainConfig):
        self.config = config
        self.rl_config = config.rl_config
        self.env_config = config.env_config
        self._debug_reset_config_state_dict = None

    @property
    def debug_reset_config_state_dict(self):
        return self._debug_reset_config_state_dict

    @debug_reset_config_state_dict.setter
    def debug_reset_config_state_dict(self, val):
        self._debug_reset_config_state_dict = val

    @staticmethod
    def get_agent_list(agent_names,
                       rainbow_config,
                       oracle_config,
                       self_play=False):
        marl_agents = {}
        learner_config = {}
        for name in agent_names:
            if name not in marl_agents:
                agent = RegisteredAgent.build_agent_instance(name, rainbow_config, oracle_config)
                marl_agents[name], conf = agent
                if name == 'rainbow_learner':
                    learner_config = conf
                    # this is monkey patched and needs some care later
                    if self_play:
                        return [agent[0] for _ in range(len(agent_names))], learner_config
        agents = []
        for name in agent_names:
            agents.append(marl_agents[name])

        return agents, learner_config

    def _run(self,
             env_config,
             num_players,
             buffer_size,
             target_update_freq,
             versus_agent_cls=None):
        # pass versus_agent_cls to set opponents
        win_rate_early_stopping = (num_players - 1) * self.env_config.starting_stack_size / self.env_config.blinds[1]
        learning_agent_ids = self.config.learning_agent_ids
        # logdir = [".", "v4", "vs_oracle", dir_suffix]
        logdir = self.config.logdir_subdirs + [f'_buffer={buffer_size}', f'_freq={target_update_freq}']
        ckpt_save_path = os.path.join(
            *logdir, f'ckpt.pt'
        )
        # agents = [str(a) for a in self.config.agents]
        # *** WARNING ***
        # NEVER CHANGE this:
        agents = [f'p{i}' for i in range(num_players)]
        # env = init_wrapped_env(**env_config)
        # obs0 = env.reset(config=None)
        num_envs = self.config.num_parallel_envs
        mc_model_ckpt_path = self.config.oracle_ckpt_dir
        self.venv, wrapped_env = make_vectorized_pettingzoo_env(
            num_envs=num_envs,
            single_env_config=env_config,
            agent_names=agents,
            mc_model_ckpt_path=mc_model_ckpt_path,
            debug_reset_config_state_dict=self._debug_reset_config_state_dict)
        self.reward_stats = Reward()
        rainbow_config = RainbowConfig(**{'device': self.rl_config.device,
                                          'load_from_ckpt': ckpt_save_path,
                                          'lr': 1e-6,
                                          'num_atoms': 51,
                                          'noisy_std': 0.1,
                                          'v_min': -6,
                                          'v_max': 6,
                                          'estimation_step': self.rl_config.n_step_lookahead,
                                          'target_update_freq': target_update_freq
                                          # training steps
                                          })
        oracle_config = OracleAgentConfig(**{'device': self.rl_config.device,
                                             'model_ckpt_path': mc_model_ckpt_path,
                                             'num_players': num_players,
                                             'model_hidden_dims': (512,)})
        marl_agents, rainbow_config = self.get_agent_list(agent_names=self.config.agents,
                                                          rainbow_config=rainbow_config,
                                                          oracle_config=oracle_config,
                                                          self_play=True)
        self.policy = MultiAgentPolicyManager(marl_agents,
                                              wrapped_env)  # policy is made from PettingZooEnv
        # policy = RainbowPolicy(**rainbow_config)
        # does not work with delayed rewards, see https://github.com/thu-ml/tianshou/issues/399
        # self.buffer = PrioritizedVectorReplayBuffer(
        #     total_size=buffer_size,
        #     buffer_num=len(self.venv),
        #     ignore_obs_next=False,  # enable for framestacking
        #     save_only_last_obs=False,  # enable for framestacking
        #     stack_num=self.rl_config.obs_stack,
        #     alpha=self.rl_config.alpha,
        #     beta=self.rl_config.beta,
        #     weight_norm=self.rl_config.weight_norm
        # )
        self.buffer = VectorReplayBuffer(buffer_size, num_envs)

        self.train_collector = Collector(self.policy, self.venv, self.buffer, exploration_noise=True)
        self.test_collector = Collector(self.policy, self.venv, exploration_noise=False)

        def train_fn(epoch, env_step, beta=self.rl_config.beta):
            try:
                # linear decay in the first 10M steps
                if env_step <= self.rl_config.eps_decay_in_first_n_steps:
                    eps = self.rl_config.eps_train - env_step / self.rl_config.eps_decay_in_first_n_steps * \
                          (self.rl_config.eps_train - self.rl_config.eps_train_final)
                else:
                    eps = self.rl_config.eps_train_final
                for aid in learning_agent_ids:
                    self.policy.policies[agents[aid]].set_eps(eps)
                if env_step % 1000 == 0:
                    logger.write("train/env_step", env_step, {"train/eps": eps})
                    logger.write("train/accumulated_reward", env_step,
                                 {"train/accumulated_reward": self.reward_stats.cumulated})
                if not self.rl_config.no_priority:
                    if env_step <= self.rl_config.beta_anneal_step:
                        beta = beta - env_step / self.rl_config.beta_anneal_step * \
                               (beta - self.rl_config.beta_final)
                    else:
                        beta = self.rl_config.beta_final
                    self.buffer.set_beta(beta)
                    if env_step % 1000 == 0:
                        logger.write("train/env_step", env_step, {"train/beta": beta})
            except Exception:
                pass

        def test_fn(epoch, env_step):
            try:
                for aid in learning_agent_ids:
                    self.policy.policies[agents[aid]].set_eps(self.rl_config.eps_test)
            except Exception:
                pass

        def save_best_fn(policy):
            for aid in learning_agent_ids:
                model_save_path = os.path.join(
                    *logdir, f'policy_{aid}.pth'
                )
                torch.save(
                    policy.policies[agents[aid]].state_dict(), model_save_path
                )

        self.early_stopping_window = 0
        self.n_early_stopping_crit = 10

        def stop_fn(mean_rewards):
            if mean_rewards >= win_rate_early_stopping:
                self.early_stopping_window += 1

            return self.early_stopping_window >= self.n_early_stopping_crit

        def reward_metric(rews):
            # The reward at index 0 is the reward relative to observer
            rew = np.mean(rews[:, learning_agent_ids[0]])
            try:
                self.reward_stats.cumulated += rew
            except OverflowError:
                # seems like we are rich
                pass
            if rew > self.reward_stats.best_reward:
                self.reward_stats.best_reward = rew
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
                'net': self.policy.state_dict(),
                'model': rainbow_config['model'].state_dict(),
                'env_step': env_step,
                'optim': rainbow_config['optim'].state_dict(),
            }, ckpt_save_path)
            return ckpt_save_path

        # test train_collector and start filling replay buffer
        self.train_collector.collect(
            n_step=self.rl_config.batch_size * num_envs)  # todo rl_config.batch_size overwritten?
        trainer = OffpolicyTrainer(policy=self.policy,
                                   train_collector=self.train_collector,
                                   test_collector=self.test_collector,
                                   max_epoch=self.rl_config.epoch,
                                   # set stop_fn for early stopping
                                   step_per_epoch=self.rl_config.step_per_epoch,
                                   # num transitions per epoch
                                   step_per_collect=self.rl_config.step_per_collect,
                                   # step_per_collect -> update network -> repeat
                                   episode_per_test=self.rl_config.episode_per_test,
                                   # games to play for one policy evaluation
                                   batch_size=self.rl_config.batch_size,
                                   update_per_step=self.rl_config.update_per_step,
                                   # fraction of steps_per_collect
                                   train_fn=train_fn,
                                   test_fn=test_fn,
                                   stop_fn=None,  # stop_fn,  # early stopping
                                   save_best_fn=save_best_fn,
                                   save_checkpoint_fn=save_checkpoint_fn,
                                   resume_from_log=self.rl_config.load_ckpt,
                                   reward_metric=reward_metric,
                                   logger=logger,
                                   verbose=True,
                                   show_progress=True,
                                   test_in_train=False
                                   # whether to test in training phase
                                   )
        result = trainer.run()
        t0 = time.time()
        pprint.pprint(result)
        print(f'took {time.time() - t0} seconds')
        # pprint.pprint(result)
        # watch()
        return self.reward_stats.best_reward

    def run(self, versus_agent_cls=None):
        """
        reset_config_state_dict: if passed, will be used to reset the environment (every episode)
        use this for testing and debugging, to initialize the train environment from specified cards and
        deck. see steinberger PokerEnv for description how to set the state dict
        """
        # environment config
        num_players = len(self.config.agents)
        env_config = dict(self.env_config)
        env_config["env_wrapper_cls"] = AugmentObservationWrapper
        env_config["stack_sizes"] = [self.env_config.starting_stack_size for _ in
                                     range(num_players)]
        env_config["multiply_by"] = 1
        env_config["agent_observation_mode"] = eval(
            self.env_config.agent_observation_mode)
        env_config.pop('starting_stack_size')

        for buffer_size in self.config.rl_config.buffer_sizes:
            for target_update_freq in self.config.rl_config.target_update_freqs:
                self._run(env_config, num_players, buffer_size, target_update_freq,
                          versus_agent_cls)


# cs = ConfigStore.instance()
# # Registering the Config class with the name 'config'.
# cs.store(name="config", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(cfg):
    """Run with `python scratch.py`, no need to overwrite in config group"""
    """
    for b in cfg.baselines:
    print(type(b))    
    <class 'str'>
    <class 'str'>
    """
    conf = TrainConfig(**cfg.training)
    TrainEval(conf).run()


if __name__ == "__main__":
    main()
