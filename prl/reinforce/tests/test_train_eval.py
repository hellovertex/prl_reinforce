import pytest


@pytest.fixture
def train_eval_runner():
    # todo define train_eval given test trainconfig
    pass


def test_train_eval_rewards_are_correct():
    # todo define state dict
    pass
# # todo: make reset config with certain cards
# # play vs 3 calling stations and test rewards
# # tests button moves
# from prl.baselines.agents.dummy_agents import DummyAgentFold, DummyAgentCall, \
#     DummyAgentAllIn
# from prl.baselines.evaluation.utils import get_reset_config, pretty_print
# from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env, \
#     make_vectorized_pettingzoo_env
# from prl.environment.Wrappers.augment import AugmentObservationWrapper
# from prl.environment.Wrappers.vectorizer import AgentObservationType
#
#
# def make_vector_env(num_players, agent_observation_mode):
#     starting_stack = 20000
#     stack_sizes = [starting_stack for _ in range(num_players)]
#     agents = [f'p{i}' for i in range(num_players)]
#     sb = 50
#     bb = 100
#     env_config = {"env_wrapper_cls": AugmentObservationWrapper,
#                   # "stack_sizes": [100, 125, 150, 175, 200, 250],
#                   "stack_sizes": stack_sizes,
#                   "multiply_by": 1,
#                   # use 100 for floats to remove decimals but we have int stacks
#                   "scale_rewards": False,  # we do this ourselves
#                   "blinds": [sb, bb],
#                   "agent_observation_mode": agent_observation_mode}
#     # env = init_wrapped_env(**env_config)
#     # obs0 = env.reset(config=None)
#     num_envs = 31
#     mc_model_ckpt_path = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/ckpt/ckpt.pt"
#     venv, wrapped_env = make_vectorized_pettingzoo_env(num_envs=num_envs,
#                                                        single_env_config=env_config,
#                                                        agent_names=agents,
#                                                        mc_model_ckpt_path=mc_model_ckpt_path)
#     return venv, wrapped_env
#
#
# mc_model_ckpt_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"
# agent_names = ["Bob_0", "Tina_1", "Alice_2", "Hans_3"]
# agent_names2 = ["Hans_3", "Bob_0", "Tina_1", "Alice_2"]
# agent_names3 = ["Alice_2", "Hans_3", "Bob_0", "Tina_1"]
#
# player_hands = ['[6s 6d]', '[9s 9d]', '[Jd Js]', '[Ks Kd]']
# board = '[6h Ts Td 9c Jc]'
# env = make_default_tianshou_env(num_players=len(agent_names),
#                                 agents=agent_names)
# venv, wrapped_env = make_vector_env(4,AgentObservationType.SEER)
#
# agents = [
#     DummyAgentAllIn,  # Bob
#     DummyAgentCall,  # Tina
#     DummyAgentFold,  # Alice
#     DummyAgentCall  # Hans
# ]
# state_dict = get_reset_config(player_hands, board)
# options = {'reset_config': state_dict}
# i = 0
# for epoch in range(4):
#     obs = env.reset(options=options)
#     obs2 = wrapped_env.reset(options=options)
#     agent_id = obs['agent_id']
#     legal_moves = obs['mask']
#     obs = obs['obs']
#     while True:
#         i = agent_names.index(agent_id)
#         action = agents[i].act(obs, legal_moves)
#         print(f'AGNET_ID = {agent_id}')
#         pretty_print(i, obs, action)
#         print(f'legal_moves = {legal_moves}')
#         obs_dict, rews, terminated, truncated, info = env.step(action)
#         obs_dict2, rews2, _, _, _ = wrapped_env.step(action)
#         assert rews == rews2
#         agent_id = obs_dict['agent_id']
#         print(f'AGENT_ID', agent_id)
#         obs = obs_dict['obs']
#         print(f'GOT REWARD {rews}')
#         if terminated:
#             print('------------------------------------')
#             print('ROUND OVER -- RESETTING ENVIRONMENT')
#             print('------------------------------------')
#             # if epoch == 0:
#             #     assert rews[1] > 0  # Tina wins with 9s 9d
#             # if epoch == 1:
#             #     assert rews[1] > 0  # Tina wins with Jd Js
#             # if epoch == 2:
#             #     assert rews[0] > 0  # Bob wins with Jd Js
#             if epoch == 0:
#                 assert rews[1] > 0  # Tina gewinnt
#             if epoch == 1:
#                 assert rews[1] > 0  # Tina wins with Jd Js
#             if epoch == 2:
#                 assert rews[
#                            0] > 0  # Bob wins with Jd Js, Hans was last to act, so it is offset 1 to bob
#             break
