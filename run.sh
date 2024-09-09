#!/bin/bash

# Define the list of alpha values
#alphas=(0.1 0.2 0.3 0.4 0.5 0.6)
alphas=(0.01)
# Path to the CSV file containing community risk values
csv_path="aggregated_weekly_risk_levels.csv"

# Loop through each alpha value and run the DQN and Myopic agents sequentially
for alpha in "${alphas[@]}"; do
#  echo "Running DQN agent training and evaluation with alpha = $alpha"
#  python3 main.py train_and_eval --alpha "$alpha" --agent_type "dqn_custom" --csv_path "$csv_path" --algorithm "dqn"
#  echo "Running PPO agent training and evaluation with alpha = $alpha"
#  python3 main.py train_and_eval --alpha "$alpha" --agent_type "ppo_custom" --csv_path "$csv_path" --algorithm "dqn"

  echo "Running A2C agent training and evaluation with alpha = $alpha"
  python3 main.py train_and_eval --alpha "$alpha" --agent_type "a2c_custom" --csv_path "$csv_path" --algorithm "dqn"

#  echo "Running Tabular Q agent training and evaluation with alpha = $alpha"
#  python3 main.py train_and_eval --alpha "$alpha" --agent_type "q_learning" --csv_path "$csv_path" --algorithm "q_learning"
# Add a delay of 10 seconds after each run
  echo "Pausing for 10 seconds before the next run..."
  sleep 5
done

echo "All training and evaluation runs completed."
