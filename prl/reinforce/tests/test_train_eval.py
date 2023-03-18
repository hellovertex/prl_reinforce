import pytest
from prl.baselines.agents.dummy_agents import DummyAgentFold, DummyAgentCall, \
    DummyAgentAllIn
from prl.baselines.agents.tianshou_agents import TianshouCallingStation, \
    TianshouAlwaysFoldAgentDummy, \
    TianshouALLInAgent
from prl.baselines.evaluation.utils import get_reset_config
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.policy import MultiAgentPolicyManager

from prl.reinforce.train_eval import TrainEval, TrainConfig

from hydra import compose, initialize
from omegaconf import DictConfig
import numpy as np
import pytest
from prl.baselines.examples.examples_tianshou_env import make_vectorized_pettingzoo_env
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.vectorizer import AgentObservationType

from prl.reinforce.train_eval import RegisteredAgent
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols


# @pytest.fixture
# def env_four_players()
@pytest.fixture
def train_eval_runner():
    # todo define train_eval given test trainconfig
    initialize(version_base=None, config_path="conf_test/training")
    cfg: DictConfig = compose('config.yaml')
    params = TrainConfig(**cfg)
    return TrainEval(params)


def test_train_eval_rewards_are_correct(train_eval_runner):
    player_hands = ['[6s 6d]', '[9s 9d]', '[Jd Js]', '[Ks Kd]']
    board = '[6h Ts Td 9c Jc]'
    # agent_names = ["Bob_0", "Tina_1", "Alice_2", "Hans_3"]
    # agent_names2 = ["Hans_3", "Bob_0", "Tina_1", "Alice_2"]
    # agent_names3 = ["Alice_2", "Hans_3", "Bob_0", "Tina_1"]
    # agents = [
    #     DummyAgentAllIn,
    #     DummyAgentCall,
    #     DummyAgentFold,
    #     DummyAgentCall
    # ]
    train_eval_runner.debug_reset_config_state_dict = get_reset_config(player_hands,
                                                                       board)
    # set breakpoints inside run and see if everything behaves as expected
    # train_eval_runner.run(versus_agent_cls=DummyAgentFold)
    train_eval_runner.run(versus_agent_cls=TianshouAlwaysFoldAgentDummy)


def get_cards(obs):
    one_hot_card_bit_range = slice(
        cols.First_player_card_0_rank_0, cols.First_player_card_1_suit_3 + 1
    )
    # observer cards are always at position 0
    return np.where(obs[one_hot_card_bit_range] == 1)


def test_collector():
    num_envs = 2
    agent_names = [RegisteredAgent.always_all_in.__name__,
                   RegisteredAgent.always_fold.__name__]
    env_config = {}
    num_players = len(agent_names)
    env_config["env_wrapper_cls"] = AugmentObservationWrapper
    env_config["stack_sizes"] = [20000 for _ in
                                 range(num_players)]
    env_config["multiply_by"] = 1
    env_config["agent_observation_mode"] = AgentObservationType.CARD_KNOWLEDGE
    env_config['scale_rewards'] = False
    env_config['blinds'] = [50, 100]
    venv, wrapped_env = make_vectorized_pettingzoo_env(
        num_envs=num_envs,
        single_env_config=env_config,
        agent_names=agent_names,
        mc_model_ckpt_path="",
        debug_reset_config_state_dict=None)
    marl_agents = [TianshouALLInAgent(), TianshouAlwaysFoldAgentDummy()]
    policy = MultiAgentPolicyManager(marl_agents,
                                     wrapped_env)
    buffer = VectorReplayBuffer(total_size=10, buffer_num=num_envs)
    buf = buffer.buffers[0]
    train_collector = Collector(policy, venv, buffer, exploration_noise=True)
    train_collector.collect(5)



    indices0 = np.where(buf.obs.agent_id == 'TianshouALLInAgent')
    indices1 = np.where(buf.obs.agent_id == 'TianshouAlwaysFoldAgentDummy')
    obs0 = buf.obs.obs[indices0]
    obs1 = buf.obs.obs[indices1]
    obs_next0 = buf.obs_next.obs[indices0]
    obs_next1 = buf.obs_next.obs[indices1]
    cards0 = [get_cards(o) for o in obs0]
    cards1 = [get_cards(o) for o in obs1]
    cards0n = [get_cards(o) for o in obs_next0]
    cards1n = [get_cards(o) for o in obs_next1]
    a = 1
    assert True
    a = 2
