# ContainerGym: A Real-World Reinforcement Learning Benchmark for Resource Allocation ♻️

[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/vwxyzjn/cleanrl)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.7-306998)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
![dependencies status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15GNDAoepHN524mFIQsieBJEohRtRt82z?usp=sharing)
## 🖊 Info
### Example render of the ContainerGym environment during evaluation
![](https://github.com/Pendu/ContainerGym_Prefinal/blob/2c3589ef8c90c77832ccc0808fc7aafa6eec1713/example.gif)

## 🏗 Folder Structure 

```txt
📦containergym
 ┣ 📂configs -> Contains .json files  used to create an environment
 ┣ 📂experiments 
    ┣📂logs -> Contains logs of training
    ┣📂results -> Graphs and results of evaluation
    ┃ ┣ 📜callbacks.py -> Callbacks for training
    ┃ ┣ 📜evaluate_agent.py -> Evaluation script
    ┃ ┗ 📜train_agent.py -> Training script
 ┣ 📂models
    ┃ ┣ 📜linear_press_models.py -> Press models for emptying containers
    ┃ ┗ 📜random_walk_models.py -> Random walk models for filling containers
 ┃ ┣ 📜env.py -> Environment module with the environment class
 ┣ 📜reward.py
 ┣ 📜project.toml
 ┗ 📜README.md   
```

## 📚 Setup

### Pre-requisites

* Python >=3.7.1,<3.10
* [Poetry 1.2.1+](https://python-poetry.org)

## 🤖 Using poetry

Clone the repository and run the following command from the root directory of the repository.

```{bash}
git clone https://github.com/Pendu/ContainerGym_Prefinal.git && cd containergym
poetry install
poetry shell

```
Run the following commands from the root directory of the repository.

### 👑 Training

```

poetry run python3 -m containergym.experiments.train_agent --config-file 1bunker1_1press.json --budget 100000 --n-steps 2048 --RL-agent PPO --n-seeds 2

```
### 📊 Evaluation

```

poetry run python3 -m containergym.experiments.evaluate_agent --config-file 1bunker1_1press.json --budget 100000 --n-steps 2048 --RL-agent PPO --n-seeds 2 --render-episode True 

```

## 🤖 Using pip

Create a virtual environment and run the following command from the root directory of the repository.

```{bash}
python3 -m venv temp_venv
source temp_venv/bin/activate
pip install -i https://test.pypi.org/simple/ containergym --extra-index-url https://pypi.org/simple

```
Run the following commands from the root directory of the repository.

### 👑 Training

```

python3 -m containergym.experiments.train_agent --config-file 1bunker1_1press.json --budget 100000 --n-steps 2048 --RL-agent PPO --n-seeds 2

```

### 📊 Evaluation

```

python3 -m containergym.experiments.evaluate_agent --config-file 1bunker1_1press.json --budget 100000 --n-steps 2048 --RL-agent PPO --n-seeds 2 --render-episode True 

```

## 🎭 Support and get involved

Feel free to ask questions. Posting in [Github Issues]( https://github.com/Pendu/ContainerGym_Prefinal/issues) and PRs are also welcome.


