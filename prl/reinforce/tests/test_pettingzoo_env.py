import numpy as np
import pytest
from prl.baselines.examples.examples_tianshou_env import make_vectorized_pettingzoo_env
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.vectorizer import AgentObservationType

from prl.reinforce.train_eval import RegisteredAgent
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols


def get_cards(obs):
    one_hot_card_bit_range = slice(
        cols.First_player_card_0_rank_0, cols.First_player_card_1_suit_3 + 1
    )
    # observer cards are always at position 0
    return np.where(obs[one_hot_card_bit_range] == 1)


# @pytest.fixture
def test_pettingzoo_env_final_step_reveals_cards_to_all_players():
    num_envs = 1
    agent_names = [RegisteredAgent.always_all_in.__name__,
                   RegisteredAgent.always_fold.__name__,
                   RegisteredAgent.always_fold.__name__ + '1',
                   RegisteredAgent.always_fold.__name__ + '2', ]
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
    all_in = 7
    fold = 0
    obs0 = wrapped_env.reset()['obs']
    obs1 = wrapped_env.step(all_in)[0]['obs']
    obs2 = wrapped_env.step(fold)[0]['obs']
    obs3 = wrapped_env.step(fold)[0]['obs']
    obs4 = wrapped_env.step(fold)[0]['obs']
    assert not np.array_equal(get_cards(obs0), get_cards(obs1))
    assert not np.array_equal(get_cards(obs1), get_cards(obs2))
    assert not np.array_equal(get_cards(obs2), get_cards(obs3))
    # Game is over, observation 3,4 are both for last player
    assert np.array_equal(get_cards(obs3), get_cards(obs4))

    # get player 0 cards
    obs_p0 = wrapped_env.env.observe(agent_names[0])['observation']
    cards = get_cards(obs_p0)
    assert np.array_equal(cards, p0_cards)

    # get player 1 cards
    obs_p1 = wrapped_env.env.observe(agent_names[1])['observation']
    cards = get_cards(obs_p1)
    assert np.array_equal(cards, p1_cards)
