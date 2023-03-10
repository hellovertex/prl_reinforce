# todo:
# 1. run self play with rainbow agent that has seer mode enabled POSTFLOP only
# 2. this will mean it learns to play perfect ranges
# 3. rl training vs random agent
import click
# todo: implement RL trainer vs Random agent
#  what do we need in terms of experimental evaluation (tables, mbb charts)
#  what do we need in terms of reproducability?
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="conf/training", config_name='config')
def train_eval(cfg):
    # python scrach.py +baselines='[random_agent]'
    # https://hydra.cc/docs/tutorials/basic/your_first_app/config_groups/
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    train_eval()
