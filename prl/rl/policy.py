import os
import pickle
from typing import Optional, Union, Any, Dict

import numpy as np
import torch
from pydantic import BaseModel
from tensorboardX import SummaryWriter
from tianshou.data import Batch, Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import C51Policy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

from env import train_envs, test_envs, n_train_envs, n_test_envs

PATH_TO_MODEL = "/home/sascha/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/cloud/aws/model.pt"


class Args(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    lr: float = 1e-3
    gamma: float = 0.99
    num_atoms: int
    v_min: float = -10
    v_max: float = 10
    n_step: int = 3
    target_update_freq: int = 320
    device: str = 'cpu'
    prioritized_replay: bool = True
    buffer_size: int = 50000
    alpha: float = 0.6
    beta: float = 0.4
    seed: int = 42
    batch_size: int = 64
    # training_num: int=8
    logdir: str = './log'
    save_interval: int = 1000
    reward_threshold: float = np.inf
    eps_train: float = 0.1
    eps_test: float = 0.00
    epoch: int = 1000
    step_per_epoch: int = 1e6
    step_per_collect: int = 1e3
    update_per_step: float = 0.125

    resume: bool = True


class FoldingBaselinePolicy(C51Policy):
    """Policy that can query baseline model and add fold probability"""

    def __init__(self, foldprob, *args, **kwargs):
        super(FoldingBaselinePolicy).__init__(*args, **kwargs)
        self._foldprob = foldprob

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            model: str = "model",
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        pass

    # todo: insert model here
    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        pass


args = Args()
net = torch.load(PATH_TO_MODEL, map_location=torch.device('cpu'))
net.eval()

optim = torch.optim.Adam(net.parameters(), lr=args.lr)
policy = FoldingBaselinePolicy(
    .5,
    net,
    optim,
    args.gamma,
    args.num_atoms,
    args.v_min,
    args.v_max,
    args.n_step,
    target_update_freq=args.target_update_freq,
).to(args.device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
train_envs.seed(args.seed)
test_envs.seed(args.seed)

buf = PrioritizedVectorReplayBuffer(
    args.buffer_size,
    buffer_num=len(train_envs),
    alpha=args.alpha,
    beta=args.beta)
train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)

""" TODO fix """
# policy.set_eps(1)
train_collector.collect(n_step=args.batch_size * n_train_envs)
# log
log_path = os.path.join(args.logdir, "c51")
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer, save_interval=args.save_interval)


def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))


def stop_fn(mean_rewards):
    return mean_rewards >= args.reward_threshold


def train_fn(epoch, env_step):
    # eps annnealing, just a demo
    if env_step <= 10000:
        policy.set_eps(args.eps_train)
    elif env_step <= 50000:
        eps = args.eps_train - (env_step - 10000) / \
              40000 * (0.9 * args.eps_train)
        policy.set_eps(eps)
    else:
        policy.set_eps(0.1 * args.eps_train)


def test_fn(epoch, env_step):
    policy.set_eps(args.eps_test)


def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt_path = os.path.join(log_path, "checkpoint.pth")
    # Example: saving by epoch num
    # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
    torch.save(
        {
            "model": policy.state_dict(),
            "optim": optim.state_dict(),
        }, ckpt_path
    )
    buffer_path = os.path.join(log_path, "train_buffer.pkl")
    pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
    return ckpt_path


if args.resume:
    # load from existing checkpoint
    print(f"Loading agent under {log_path}")
    ckpt_path = os.path.join(log_path, "checkpoint.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        policy.load_state_dict(checkpoint["model"])
        policy.optim.load_state_dict(checkpoint["optim"])
        print("Successfully restore policy and optim.")
    else:
        print("Fail to restore policy and optim.")
    buffer_path = os.path.join(log_path, "train_buffer.pkl")
    if os.path.exists(buffer_path):
        train_collector.buffer = pickle.load(open(buffer_path, "rb"))
        print("Successfully restore buffer.")
    else:
        print("Fail to restore buffer.")

# trainer
result = OffpolicyTrainer(
    policy,
    train_collector,
    test_collector,
    args.epoch,
    args.step_per_epoch,
    args.step_per_collect,
    n_test_envs,
    args.batch_size,
    update_per_step=args.update_per_step,
    train_fn=train_fn,
    test_fn=test_fn,
    stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    logger=logger,
    resume_from_log=args.resume,
    save_checkpoint_fn=save_checkpoint_fn,
).run()
assert stop_fn(result["best_reward"])

# if __name__ == "__main__":
#     pprint.pprint(result)
#     # Let's watch its performance!
#     env = gym.make(args.task)
#     policy.eval()
#     policy.set_eps(args.eps_test)
#     collector = Collector(policy, env)
#     result = collector.collect(n_episode=1, render=args.render)
#     rews, lens = result["rews"], result["lens"]
#     print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
