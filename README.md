# ContainerGym: A Real-World Reinforcement Learning Benchmark for Resource Allocation ♻️

![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.7-306998)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)


## 🏗 Folder Structure 

```txt
📦containergym
 ┣ 📂configs -> Contains .json files  used to create an environment
 ┣ 📂playground -> To run local experiments
    ┣ 📜callbacks.py -> Callbacks for training
    ┣ 📜evaluate_agent.py -> Evaluation script
    ┗ 📜train_agent.py -> Training script
 ┣ 📂models
   ┣ 📜linear_press_models.py -> Press models for emptying containers
   ┗ 📜random_walk_models.py -> Random walk models for filling containers
 ┃ 📜env.py -> Environment module with the environment class
 ┣ 📜reward.py
 📦experiments -> To reproduce experiments from the paper
  ┣📂logs_best_seeds -> Contains logs of training
  ┣ 📜reproduce_results_paper.py
  ┣ 📜rule_based_agent.py
 ┣ 📜project.toml
 ┗ 📜README.md   
```

## 📚 Setup

### Pre-requisites (Important)

* Python >=3.9.0,<3.10
* optional guide for the user: If existing python version on Linux based system is not meeting the pre-requisites. 
* Use pyenv for installing a new python version 3.9.0 system-wide

```{bash}
sudo apt install curl
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv install 3.9.0
pyenv global 3.9.0
pyenv local 3.9.0
```

## 🤖 Installation

Clone the repository and run the following.

```{bash}
git clone https://anonymous.4open.science/r/ContainerGym
cd ContainerGym_Prefinal
```

Create a virtual environment and run the following commands

```{bash}
python3 -m venv temp_venv
source temp_venv/bin/activate
pip install -r requirements.txt
```

## Try it out locally 
### 👑 Training

```
 python3 -m containergym.playground.train_agent --config-file 1container_1press.json --budget 100000 --n-steps 2048 --RL-agent PPO --n-seeds 1
```
### 📊 Evaluation

```
python3 -m containergym.playground.evaluate_agent --config-file 1container_1press.json --budget 100000 --n-steps 2048 --RL-agent PPO --n-seeds 1 --render-episode True 
```

## Reproduce results from the paper
```
cd experiments
python reproduce_results_paper.py
```



