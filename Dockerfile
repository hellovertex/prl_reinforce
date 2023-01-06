FROM ubuntu


RUN apt-get update -y && apt-get install -y
RUN apt-get install python3-pip -y

# Dependencies for building the c++ hand evaluator library
RUN apt-get install cmake -y
RUN apt-get install build-essential -y
RUN apt-get install make -y
RUN apt-get install libpython3.10-dev -y

# Get relevant python packages from github
# # Option 1) Using COPY
#COPY . /ml/
#WORKDIR ml

# # Option 2) Using git clone
RUN mkdir /ml/
WORKDIR ml
RUN apt-get install git -y
RUN cd /ml/ && git clone --recurse-submodules https://github.com/hellovertex/prl_environment
RUN cd prl_environment && git submodule update --recursive --remote
RUN cd /ml/ && git clone --recurse-submodules https://github.com/hellovertex/prl_baselines
RUN cd prl_baselines && git submodule update --recursive --remote
RUN cd /ml/ && git clone --recurse-submodules https://github.com/hellovertex/prl_reinforce
RUN cd prl_reinforce && git submodule update --recursive --remote && cd ..

# Install Poker Environment
RUN pip install requests  # for some reason cannot be installed from requirements.txt
RUN cd /ml/prl_environment && pip install .

# Install Poker Baseline Agents
RUN cd /ml/prl_baselines && pip install .
# ... with c++ hand evaluator library
RUN cd /ml/prl_baselines/prl/baselines/cpp_hand_evaluator/cpp
RUN cmake . && make

# Install Poker Reinforcement Learning package
RUN cd /ml/prl_reinforce && pip install .

RUN export ALGO_CKPT_DIR=/ml/prl_reinforce/data/rllib_ckpts
RUN export PRL_BASELINE_MODEL_PATH=/ml/prl_reinforce/data/baseline_model_ckpt.pt

ENTRYPOINT /ml/prl_reinforce/prl/reinforce/train_using_rllib/example.py
# CMD python /ml/prl_reinforce/prl/reinforce/train_using_rllib/example.py
