import gin
from ray.rllib.algorithms import AlgorithmConfig
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
            # algorithm_config: AlgorithmConfigDict,
            algorithm_config: AlgorithmConfig,
            algo_ckpt_dir,
            ckpt_interval
            ):
        # algorithm_class(algorithm_config)
        algo = algorithm_config.build()
        # maybe load from checkpoint
        try:
            algo.from_checkpoint(algo_ckpt_dir)
        except ValueError as e:
            # if checkpoint does not exist, we start from scratch
            print('No Rllib Algorithm Checkpoint exists at given directory, '
                  'we start training from scratch. Original Error Message was ', e)
        import gc
        import psutil
        # how many steps
        for i in range(1000000):
            results = algo.train()
            if psutil.virtual_memory().percent >= 50:
                gc.collect()
            if (i + 1) % ckpt_interval:
                algo.save_checkpoint(algo_ckpt_dir)
