import gin
from ray.rllib.algorithms.simple_q import SimpleQ
from ray.rllib.utils.typing import AlgorithmConfigDict

RAINBOW_POLICY = "SimpleQ"
BASELINE_POLICY = "StakeImitation"
DistributedRainbow = SimpleQ

BASELINE_AGENT = "Baseline"
TRAINABLE_AGENT = "Trainable"


class TrainRunner:
    def __init__(self):
        pass

    @gin.configurable
    def run_from_gin_configfile(self):
        pass

    def run(self,
            algorithm_class,  # our custom dict, NOT rllib EnvContext dictionary
            algorithm_config: AlgorithmConfigDict,
            algo_ckpt_dir,
            ckpt_interval
            ):
        algo = algorithm_class(algorithm_config)
        # maybe load from checkpoint
        try:
            algo.from_checkpoint(algo_ckpt_dir)
        except ValueError as e:
            # if checkpoint does not exist, we start from scratch
            print('No Rllib Algorithm Checkpoint exists at given directory, '
                  'we start training from scratch. Original Error Message was ', e)

        # how many steps
        for i in range(10000000000000):
            results = algo.train()
            if (i + 1) % ckpt_interval:
                algo.save_checkpoint(algo_ckpt_dir)
