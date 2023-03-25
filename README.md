# An RL benchmark framework based on real world system

This branch contains a version of the RL benchmark environment which will be published alongside the benchmark paper.

# Folder Structure
The **configs** folder contains .json files which can be used to create an environment. These files currently hold the parameters we extracted from Sutco's data, and should be modified before release!

The **models** folder contains the press models for bunker emptying and random walk models for bunker filling.

**env.py** contains the environment, and **reward.py** contains a class that can be parameterized and be used to calculate rewards.

# Quick Start
 
To configure your environment you will need Anaconda, the Python Distribution.

The instructions for installing Anaconda can be found [here](https://docs.anaconda.com/anaconda/install/)

Once Anaconda is installed you should have `conda` executable in your environment path.

Anaconda provides a concept called environments which allow us to have different dependencies based on what we're working on. The documentation is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Instructions: 

```{bash}
conda env create -f environment.yml
```


# Link to shared folder 
https://drive.google.com/drive/folders/1j1VAmhpyXU7DKTFTjCtpHqZcMrzNabAV?usp=sharing

# Link to paperpile 
https://paperpilegit
zim.com/shared/QFCh51

## Usage:

Execute from the experiments/ folder. 

### Training

```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9

```

```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent PPO --n-seeds 5

```

```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent TRPO --n-seeds 9

```
```

python train_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent A2C --n-seeds 5

```

### Evaluation

```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9 --inf-eplen 600

```
```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent PPO --n-seeds 5 --inf-eplen 600

```

```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent TRPO --n-seeds 9 --inf-eplen 600

```
```

python evaluate_agent.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent A2C --n-seeds 5 --inf-eplen 600

```

### Plot reward curves (training) for an Agent for each seed in different plots

```

python plot_smoothened_reward.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9

```

### Plot reward curve (training) for an Agent for all seeds in a single plot

```

python plot_smoothened_reward_all_seeds.py --config-file 5bunkers_2presses.json --budget 2000000 --n-steps 6144 --RL-agent DQN --n-seeds 9

```

