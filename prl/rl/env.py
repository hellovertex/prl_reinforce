# create augmented environment
# wrap with BaseVectorEnv inject into collector
# set env_args such that new starting stacks are used
from functools import partial

from prl.environment.Wrappers.prl_wrappers import AugmentObservationWrapper
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from tianshou.env import SubprocVectorEnv

DEFAULT_STARTING_STACK_SIZE = 20000
n_players = 6
starting_stack_sizes = [DEFAULT_STARTING_STACK_SIZE for _ in range(2)]
n_train_envs = 5
n_test_envs = 3
def make_wrapped_env(n_players, starting_stack_sizes):
    # set env_args such that new starting stacks are used
    args = NoLimitHoldem.ARGS_CLS(n_seats=n_players,
                                  starting_stack_sizes_list=starting_stack_sizes,
                                  use_simplified_headsup_obs=False)
    env = NoLimitHoldem(is_evaluating=True,
                        env_args=args,
                        lut_holder=NoLimitHoldem.get_lut_holder())
    env = AugmentObservationWrapper(env)
    env.overwrite_args(args)
    return env

env_gen = partial(make_wrapped_env, n_players=n_players,
                  starting_stack_sizes=starting_stack_sizes)

train_envs = SubprocVectorEnv(env_fns=[env_gen for _ in range(n_train_envs)])
test_envs  = SubprocVectorEnv(env_fns=[env_gen for _ in range(n_test_envs)])