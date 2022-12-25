# prl_reinforce

## Installation

----

Prerequisites: Make sure 
- `htpps://github.com/hellovertex/prl_environment.git` and 
- `htpps://github.com/hellovertex/prl_baselines.git` 

are installed (see their `README`s -- clone them and install via `pip install .`). Then

1. `git clone https://github.com/hellovertex/prl_reinforce.git`
2. `cd prl_reinforce`
3. `pip install -e .`  `# use -e optionally for development`

## Usage

----

To execute an example distributed training run using ray and an rllib vectorized-PokerEnv
you want to 
- run `python prl.reinforce.train_using_rllib.example.py`

Before doing that, please set output path inside `prl/reinforce/config.gin`.


## Development

----

### Remotely debug on EC2 VM instance using PyCharm Professional 
Prerequisites: 
[Connect to VM via ssh](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)

TL;DR -- consider setting an alias inside .bashrc:

`alias r1='ssh -i <PATH_TO_.pem_FILE> ubuntu@<INSTANCE-PUBLIC-IPv4-ADRESS>'`

If you cant see Public IPv4 Addresses, goto VM Actions -> Networking -> Manage IP addresses

