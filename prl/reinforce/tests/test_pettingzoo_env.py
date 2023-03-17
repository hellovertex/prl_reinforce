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
    all_in = 7
    fold = 0
    action = np.zeros(1)
    # get player 0 cards
    observation = wrapped_env.reset()
    obs = observation[0]['obs']
    p0_cards = get_cards(obs)
    # get player 1 cards
    observation = wrapped_env.step(all_in)
    obs = observation[0]['obs']
    p1_cards = get_cards(obs)
    assert p0_cards != p1_cards
    # end game
    observation = wrapped_env.step(fold)
    obs = observation[0]['obs']
    cards = get_cards(obs)
    assert cards == p1_cards

    # get player 0 cards
    obs_p0 = wrapped_env.env.observe(agent_names[0])['observation']
    cards = get_cards(obs_p0)
    assert cards == p0_cards
    # get player 1 cards
    obs_p1 = wrapped_env.env.observe(agent_names[1])['observation']
    cards = get_cards(obs_p1)
    assert cards == p1_cards
