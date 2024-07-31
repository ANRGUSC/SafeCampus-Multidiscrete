#!/bin/bash

# Define the list of alpha values
alphas=(0.2 0.4 0.5 0.8)

# Export main.py so it is available to subshells
export -f main.py

# Run tasks in parallel
printf "%s\n" "${alphas[@]}" | xargs -I {} -P 8 bash -c 'echo "Running training with alpha = {}"; python3 main.py train --alpha {} --agent_type dqn_custom'

echo "All training runs completed."