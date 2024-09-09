import os
os.environ['WANDB_MODE'] = 'offline'
import yaml
import gymnasium as gym
import numpy as np
import wandb
import argparse
from pathlib import Path
from campus_gym.envs.campus_gym_env import CampusGymEnv
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
# Set W&B to offline mode
import random
from datetime import datetime


# Generate a random name


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_environment(shared_config_path, read_community_risk_from_csv=False, csv_path=None,
                           algorithm='q_learning', mode='train'):
    shared_config = load_config(shared_config_path)
    env = CampusGymEnv(read_community_risk_from_csv=read_community_risk_from_csv, csv_path=csv_path,
                       algorithm=algorithm, mode=mode)
    return env, shared_config



def safe_wandb_init(**kwargs):
    try:
        return wandb.init(**kwargs)
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return None

def safe_wandb_log(metrics):
    try:
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception as e:
        print(f"Failed to log metrics to wandb: {e}")

def run_training_and_evaluation(env, shared_config_path, alpha, agent_type, algorithm, csv_path):
    try:
        timestamp = datetime.now().strftime("%H%M%S")
        run_name = f"{agent_type}_{timestamp}_{alpha}_{5}"

        print("Starting training phase...")
        env_train, _ = initialize_environment(shared_config_path, read_community_risk_from_csv=False,
                                              algorithm=algorithm, mode='train')
        agent = run_training(env_train, shared_config_path, alpha, agent_type, algorithm, run_name)

        print(f"Training complete. Starting evaluation phase using CSV: {csv_path}")
        env_eval, _ = initialize_environment(shared_config_path, read_community_risk_from_csv=True, csv_path=csv_path,
                                             algorithm=algorithm, mode='eval')
        run_evaluation(env_eval, shared_config_path, agent_type, alpha, run_name, algorithm, csv_path)

    except Exception as e:
        print(f"An error occurred: {e}")

def run_training(env, shared_config_path, alpha, agent_type, algorithm, run_name):
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)

    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name, shared_config_path=shared_config_path, agent_config_path=agent_config_path)

    agent.train(alpha)

    filename = str(f'run_names_{agent_type}.txt')
    with open(filename, 'a') as file:
        file.write(run_name + '\n')

    print("Done Training with alpha: ", alpha, "agent_type: ", agent_type, "algorithm: ", algorithm, "run_name: ", run_name)
    return agent


def format_agent_class_name(agent_type):
    special_acronyms = {
        'offppo': 'OffPPO',
        'ppo': 'PPO',
        'dqn': 'DQN',
        'a2c': 'A2C',
        'ddpg': 'DDPG',
        'sac': 'SAC',
        'td3': 'TD3',
    }
    parts = agent_type.split('_')
    formatted_parts = [special_acronyms.get(part, part.capitalize()) for part in parts]
    return ''.join(formatted_parts) + 'Agent'


# def run_training(env, shared_config_path, alpha, agent_type, algorithm, run_name):
#     # Remove the wandb.init() call from here since it's now in run_training_and_evaluation
#
#     agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
#     agent_config = load_config(agent_config_path)
#     wandb.config.update(agent_config)
#     wandb.config.update({'alpha': alpha, 'algorithm': algorithm, 'run_name': run_name})
#
#     AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
#     AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
#     agent = AgentClass(env, run_name, shared_config_path=shared_config_path, agent_config_path=agent_config_path)
#
#     agent.train(alpha)
#
#     filename = str(f'run_names_{agent_type}.txt')
#     with open(filename, 'a') as file:
#         file.write(run_name + '\n')
#
#     print("Done Training with alpha: ", alpha, "agent_type: ", agent_type, "algorithm: ", algorithm, "run_name: ",
#           run_name)
#     return agent

def run_sweep(shared_config_path, agent_type, algorithm):
    # Initialize wandb for this sweep run
    run = wandb.init()
    config = wandb.config

    # Initialize the environment for this specific run
    env, _ = initialize_environment(shared_config_path, algorithm=algorithm, mode='train')

    # Generate a unique run name
    run_name = f"{wandb.run.name}-{config.alpha}"

    # Run the training
    run_training(env, shared_config_path, config.alpha, agent_type, algorithm, run_name)

    # Close the environment
    env.close()

def run_optuna(env, shared_config_path, agent_type):
    shared_config = load_config(shared_config_path)
    optuna_config_path = os.path.join('config', 'optuna_config.yaml')
    optuna_config = load_config(optuna_config_path)

    def objective(trial):
        wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'], reinit=True)

        config = {'agent': {}}  # Ensure 'agent' key exists
        for param, param_config in optuna_config['parameters'].items():
            if param_config['type'] == 'float':
                config['agent'][param] = trial.suggest_float(param, param_config['min'], param_config['max'])
            elif param_config['type'] == 'int':
                config['agent'][param] = trial.suggest_int(param, param_config['min'], param_config['max'])
            elif param_config['type'] == 'categorical':
                config['agent'][param] = trial.suggest_categorical(param, param_config['values'])

        wandb.config.update(config['agent'])

        tr_name = wandb.run.name
        agent_name = f"optuna_{tr_name}"

        AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
        AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
        agent = AgentClass(env, agent_name, shared_config_path=shared_config_path, override_config=config)

        agent.train(config['agent']['alpha'])

        final_performance = agent.get_final_performance()

        wandb.finish()

        return final_performance

    study = optuna.create_study(direction=optuna_config.get('direction', 'maximize'))
    study.optimize(objective, n_trials=optuna_config.get('n_trials', 20))

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_contour(study)
    fig4 = plot_slice(study)

    fig1.write_html(os.path.join('optuna_runs', "optuna_optimization_history.html"))
    fig2.write_html(os.path.join('optuna_runs', "optuna_param_importances.html"))
    fig3.write_html(os.path.join('optuna_runs', "optuna_contour.html"))
    fig4.write_html(os.path.join('optuna_runs', "optuna_slice.html"))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def run_evaluation(env, shared_config_path, agent_type, alpha, run_name, algorithm, csv_path=None):
    print("Running Evaluation...")
    print("csv_path: ", csv_path)
    print("algorithm: ", algorithm)

    # Load agent configuration
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)

    # Initialize agent
    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name, shared_config_path=shared_config_path, agent_config_path=agent_config_path,
                       csv_path=csv_path)

    # Run the evaluation
    total_rewards = agent.evaluate(run_name=run_name, alpha=alpha, csv_path=csv_path)

    # Log results
    print(f"Total Reward for {agent_type} agent using {algorithm} algorithm: {sum(total_rewards)}")


    # Save results to a file
    results_dir = os.path.join('results', agent_type, run_name)
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'evaluation_results_{algorithm}_{alpha}.txt')

    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {agent_type} agent using {algorithm} algorithm\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Total Reward: {sum(total_rewards)}\n")
        f.write("Evaluation Metrics:\n")


    print(f"Evaluation results saved to {results_file}")

    return total_rewards


def run_evaluation_random(env, shared_config_path, agent_type, alpha, run_name, algorithm):
    print("Running Random Evaluation...")
    print("algorithm: ", algorithm)

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name, shared_config_path=shared_config_path, agent_config_path=os.path.join('config', f'config_{agent_type}.yaml'))

    test_episodes = 4
    evaluation_metrics = agent.test_baseline_random(test_episodes, alpha)

    print(f"Evaluation Metrics for random agent using {algorithm} algorithm:", evaluation_metrics)


def run_multiple_runs(env, shared_config_path, agent_type, alpha_t, beta_t, num_runs):
    shared_config = load_config(shared_config_path)
    wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])

    tr_name = wandb.run.name
    agent_name = f"multi_{tr_name}_{alpha_t}_{beta_t}_{num_runs}"

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)
    wandb.config.update(agent_config)
    wandb.config.update({'alpha_t': alpha_t, 'beta_t': beta_t, 'num_runs': num_runs})

    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, agent_name, shared_config_path=shared_config_path, agent_config_path=agent_config_path)

    agent.multiple_runs(num_runs, alpha_t, beta_t)

    print("Done Multiple Runs with alpha_t: ", alpha_t, "beta_t: ", beta_t, "agent_type: ", agent_type, "agent_name: ", agent_name)
    return agent_name


def main():
    parser = argparse.ArgumentParser(description='Run training, evaluation, or combined training and evaluation.')
    parser.add_argument('mode', choices=['train', 'eval', 'sweep', 'multi', 'optuna', 'train_and_eval'],
                        help='Mode to run the script in.')
    parser.add_argument('--alpha', type=float, default=0.3, help='Reward parameter alpha.')
    parser.add_argument('--alpha_t', type=float, default=0.05, help='Alpha value for tolerance interval.')
    parser.add_argument('--beta_t', type=float, default=0.9, help='Beta value for tolerance interval.')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs for tolerance interval.')
    parser.add_argument('--agent_type', default='q_learning', help='Type of agent to use.')
    parser.add_argument('--run_name', default=None, help='Unique name for the training run or evaluation.')
    parser.add_argument('--read_from_csv', action='store_true', help='Read community risk values from CSV.')
    parser.add_argument('--csv_path', default=None, help='Path to the CSV file containing community risk values.')
    parser.add_argument('--algorithm', choices=['q_learning', 'dqn'], default='q_learning',
                        help='Algorithm to use (q_learning or dqn)')

    global args
    args = parser.parse_args()

    shared_config_path = os.path.join('config', 'config_shared.yaml')

    if args.mode == 'train_and_eval':
        if not args.csv_path:
            raise ValueError("CSV path must be provided for train_and_eval mode")
        run_training_and_evaluation(None, shared_config_path, args.alpha, args.agent_type, args.algorithm,
                                    args.csv_path)
    elif args.mode == 'train':
        env, _ = initialize_environment(shared_config_path, algorithm=args.algorithm, mode='train')

        shared_config = load_config(shared_config_path)
        timestamp = datetime.now().strftime("%H%M%S")
        run_name = f"{args.agent_type}_{timestamp}_{args.alpha}_{5}"
        run = safe_wandb_init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'],
                              name=run_name, mode="offline")

        run_training(env, shared_config_path, args.alpha, args.agent_type, args.algorithm, run_name)
    elif args.mode == 'eval':
        if not args.csv_path or not args.run_name:
            raise ValueError("CSV path and run name must be provided for eval mode")
        env, _ = initialize_environment(shared_config_path, read_community_risk_from_csv=True, csv_path=args.csv_path,
                                        algorithm=args.algorithm, mode='eval')
        run_evaluation(env, shared_config_path, args.agent_type, args.alpha, args.run_name, args.algorithm,
                       args.csv_path)
    elif args.mode == 'sweep':
        sweep_config_path = os.path.join('config', 'sweep.yaml')
        sweep_config = load_config(sweep_config_path)
        shared_config = load_config(shared_config_path)
        sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'],
                               entity=shared_config['wandb']['entity'])
        wandb.agent(sweep_id, function=lambda: run_sweep(shared_config_path, args.agent_type, args.algorithm))
    elif args.mode == 'multi':
        env, _ = initialize_environment(shared_config_path, algorithm=args.algorithm, mode='train')
        run_multiple_runs(env, shared_config_path, args.agent_type, args.alpha_t, args.beta_t, args.num_runs)
    elif args.mode == 'optuna':
        env, _ = initialize_environment(shared_config_path, algorithm=args.algorithm, mode='train')
        run_optuna(env, shared_config_path, args.agent_type)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == '__main__':
    main()
