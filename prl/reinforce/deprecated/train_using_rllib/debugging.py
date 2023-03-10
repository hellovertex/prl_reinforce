"""Used for debugging ray applications - See https://docs.ray.io/en/latest/ray-observability/ray-debugging.html"""
import gin

from prl.reinforce.train_using_rllib.example import run

if __name__ == '__main__':
    gin.parse_config_file('./gin_configs/config_example.gin')
    run()