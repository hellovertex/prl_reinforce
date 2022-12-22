# prl_reinforce

## Installation
Prerequisites: Make sure 
- `htpps://github.com/hellovertex/prl_environment.git` and 
- `htpps://github.com/hellovertex/prl_baselines.git` 

are installed (see their `README`s -- clone them and install via `pip install .`). Then

1. `git clone https://github.com/hellovertex/prl_reinforce.git`
2. `cd prl_reinforce`
3. `pip install -e .`  `# use -e optionally for development`

## Run example 
To execute an example distributed training run using ray and an rllib vectorized-PokerEnv
you want to 
- run `python prl.reinforce.train_using_rllib.example.py`

Before doing that, please set output path inside `prl/reinforce/config.gin`.
