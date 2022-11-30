import enum

import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from prl.baselines.supervised_learning.models.nn_model import MLP
from prl.environment.multi_agent.utils import make_multi_agent_env


class TrainableModelType(enum.IntEnum):
    CUSTOM_TORCH_MLP = 0
    MLP_2x512 = 10
    RANDOM_FOREST = 20


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.internal_model = FullyConnectedNetwork(obs_space,
                                                    action_space,
                                                    num_outputs,
                                                    model_config,
                                                    name + '_internal')

    def forward(self, input_dict, state, seq_lens):
        if isinstance(input_dict, dict):
            return self.internal_model(input_dict['obs'])
        # rllib preprocessor api has already flattened observation dictionary
        # --> first three bits after flattening correspond to legal moves mask
        return self.internal_model(input_dict['obs'][:, 3:])


ModelCatalog.register_custom_model(f"{TrainableModelType.CUSTOM_TORCH_MLP.name}", CustomTorchModel)
