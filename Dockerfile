FROM ubuntu
#FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker
ENV PATH="/opt/ml/code/code:${PATH}"

RUN apt-get update -y && apt-get install -y
RUN apt-get install python3-pip -y
RUN pip install sagemaker-training

# Dependencies for building the c++ hand evaluator library
RUN apt-get install cmake -y
RUN apt-get install build-essential -y
RUN apt-get install make -y
RUN apt-get install libpython3.10-dev -y

# Get relevant python packages from github
# # Option 1) Using COPY
#COPY . /opt/ml/code/
#WORKDIR ml

# # Option 2) Using git clone
RUN mkdir -p /opt/ml/code/
WORKDIR /opt/ml/code/
RUN apt-get install git -y
RUN cd /opt/ml/code/ && git clone --recurse-submodules https://github.com/hellovertex/prl_environment
RUN cd prl_environment && git submodule update --recursive --remote
RUN cd /opt/ml/code/ && git clone --recurse-submodules https://github.com/hellovertex/prl_baselines
RUN cd prl_baselines && git submodule update --recursive --remote
RUN cd /opt/ml/code/ && git clone --recurse-submodules https://github.com/hellovertex/prl_reinforce
RUN cd prl_reinforce && git submodule update --recursive --remote && cd ..

# Install Poker Environment
RUN pip install requests  # for some reason cannot be installed from requirements.txt
RUN cd /opt/ml/code/prl_environment && pip install .

# Install Poker Baseline Agents
RUN cd /opt/ml/code/prl_baselines && pip install .
# ... with c++ hand evaluator library
RUN cd /opt/ml/code/prl_baselines/prl/baselines/cpp_hand_evaluator/cpp && cmake . && make

# Install Poker Reinforcement Learning package
RUN cd /opt/ml/code/prl_reinforce && pip install .

RUN export ALGO_CKPT_DIR=/opt/ml/code/prl_reinforce/data/rllib_ckpts
RUN export PRL_BASELINE_MODEL_PATH=/opt/ml/code/prl_reinforce/data/baseline_model_ckpt.pt

RUN pip install sagemaker-training
# ENTRYPOINT /opt/ml/code/prl_reinforce/prl/reinforce/train_using_rllib/example_docker_sagemaker.py
ENV SAGEMAKER_PROGRAM /opt/ml/code/prl_reinforce/prl/reinforce/train_using_rllib/example_docker_sagemaker.py
# CMD python /opt/ml/code/prl_reinforce/prl/reinforce/train_using_rllib/example.py
CMD python /opt/ml/code/prl_reinforce/prl/reinforce/train_using_rllib/example_docker_sagemaker.py
