#!/bin/bash

# Define the list of alpha values
alphas=(0.2 0.3 0.5 0.8)

# Export main.py so it is available to subshells
export -f main.py

# Run tasks in parallel
printf "%s\n" "${alphas[@]}" | xargs -I {} -P 8 bash -c 'echo "Running training with alpha = {}"; python3 main.py train --alpha {}'

echo "All training runs completed."

# Training Commands

# Q-learning Training (without CSV)
python your_script.py train --agent_type q_learning --algorithm q_learning --alpha 0.8

# Q-learning Training (with CSV)
python your_script.py train --agent_type q_learning --algorithm q_learning --alpha 0.8 --read_from_csv --csv_path path/to/your/training_data.csv

# DQN Training (without CSV)
python your_script.py train --agent_type dqn --algorithm dqn --alpha 0.8

# DQN Training (with CSV)
python your_script.py train --agent_type dqn --algorithm dqn --alpha 0.8 --read_from_csv --csv_path path/to/your/training_data.csv

# Evaluation Commands

# Q-learning Evaluation (without CSV)
python your_script.py eval --agent_type q_learning --algorithm q_learning --alpha 0.8 --run_name q_learning_run_001

# Q-learning Evaluation (with CSV)
python your_script.py eval --agent_type q_learning --algorithm q_learning --alpha 0.8 --run_name q_learning_run_001 --read_from_csv --csv_path path/to/your/eval_data.csv

# DQN Evaluation (without CSV)
python your_script.py eval --agent_type dqn --algorithm dqn --alpha 0.8 --run_name dqn_run_001

# DQN Evaluation (with CSV)
python your_script.py eval --agent_type dqn --algorithm dqn --alpha 0.8 --run_name dqn_run_001 --read_from_csv --csv_path path/to/your/eval_data.csv

# Random Agent Evaluation (Q-learning environment, without CSV)
python your_script.py random --agent_type q_learning --algorithm q_learning --alpha 0.8 --run_name random_q_learning_001

# Random Agent Evaluation (Q-learning environment, with CSV)
python your_script.py random --agent_type q_learning --algorithm q_learning --alpha 0.8 --run_name random_q_learning_001 --read_from_csv --csv_path path/to/your/eval_data.csv

# Random Agent Evaluation (DQN environment, without CSV)
python your_script.py random --agent_type dqn --algorithm dqn --alpha 0.8 --run_name random_dqn_001

# Random Agent Evaluation (DQN environment, with CSV)
python your_script.py random --agent_type dqn --algorithm dqn --alpha 0.8 --run_name random_dqn_001 --read_from_csv --csv_path path/to/your/eval_data.csv