# SafeCampus-Multidiscrete

This project is part of the main SafeCampus project is designed to train and evaluate reinforcement learning agents in a custom Gym environment for campus scenarios. The agents are implemented using different RL algorithms, such as Q-Learning and DQN.

## Installation

### Prerequisites

1. Python 3.8 or higher
2. [pip](https://pip.pypa.io/en/stable/)
3. [virtualenv](https://virtualenv.pypa.io/en/latest/)

### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/SafeCampus-Multidiscrete.git
    cd SafeCampus-Multidiscrete
    ```

2. Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Install the custom Gym environment:

    ```bash
    cd campus_gym
    pip install -e .
    cd ..
    ```

## Configuration

Configuration files for the shared settings and specific agents are located in the `config` directory.

### Shared Configuration

The shared configuration is specified in `config/config_shared.yaml`. It includes settings that are common across different agents, such as environment details and logging configurations.

### Agent-Specific Configuration

Agent-specific configurations are specified in separate YAML files, such as `config/config_q_learning.yaml` for Q-Learning and `config/config_dqn.yaml` for DQN. These files contain settings specific to each agent, including hyperparameters.

## Running the Project

### Training an Agent

To train an agent, use the following command:

```bash
python main.py train --alpha 0.5 --agent_type q_learning
```
### Evaluating an Agent
To evaluate a trained agent, use the following command:

```bash
python main.py eval --alpha 0.5 --agent_type q_learning --run_name your_run_name
```
or for DQN:
```bash
python main.py eval --alpha 0.5 --agent_type dqn --run_name your_run_name
```
### Running Multiple Training Runs
To run multiple training runs and calculate tolerance intervals, use:
```bash
python main.py multi --alpha_t 0.05 --beta_t 0.9 --num_runs 5 --agent_type q_learning
```
or for DQN:
```bash
python main.py multi --alpha_t 0.05 --beta_t 0.9 --num_runs 5 --agent_type dqn
```
### Hyperparameter Optimization with Optuna
To perform hyperparameter optimization using Optuna, use:
```bash
python main.py optuna --agent_type q_learning
```
or for DQN:
```bash
python main.py optuna --agent_type dqn
```
