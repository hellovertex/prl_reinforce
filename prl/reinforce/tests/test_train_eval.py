import pytest
from prl.baselines.agents.dummy_agents import DummyAgentFold, DummyAgentCall, DummyAgentAllIn
from prl.baselines.agents.tianshou_agents import TianshouCallingStation, TianshouAlwaysFoldAgentDummy
from prl.baselines.evaluation.utils import get_reset_config

from prl.reinforce.train_eval import TrainEval, TrainConfig

from hydra import compose, initialize
from omegaconf import DictConfig


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
    train_eval_runner.debug_reset_config_state_dict = get_reset_config(player_hands, board)
    # set breakpoints inside run and see if everything behaves as expected
    # train_eval_runner.run(versus_agent_cls=DummyAgentFold)
    train_eval_runner.run(versus_agent_cls=TianshouAlwaysFoldAgentDummy)