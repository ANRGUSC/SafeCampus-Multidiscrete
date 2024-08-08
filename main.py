import os
import yaml
import gymnasium as gym
import numpy as np
import wandb
import argparse
from pathlib import Path
from campus_gym.envs.campus_gym_env import CampusGymEnv
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def initialize_environment(shared_config_path, read_community_risk_from_csv=False, csv_path=None):
    shared_config = load_config(shared_config_path)
    env = CampusGymEnv(read_community_risk_from_csv=read_community_risk_from_csv, csv_path=csv_path)
    return env, shared_config

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

def run_training(env, shared_config_path, alpha, agent_type, is_sweep=False):
    if not is_sweep:
        shared_config = load_config(shared_config_path)
        wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])

    if wandb.run is None:
        raise RuntimeError("wandb run has not been initialized. Please make sure wandb.init() is called before run_training.")

    tr_name = wandb.run.name + '_' + str(alpha)
    agent_name = f"sweep_{tr_name}" if is_sweep else str(tr_name)

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    agent_config = load_config(agent_config_path)
    wandb.config.update(agent_config)
    wandb.config.update({'alpha': alpha})
    effective_alpha = wandb.config.alpha if is_sweep else alpha
    env.alpha = effective_alpha

    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    if is_sweep:
        agent = AgentClass(env, agent_name, shared_config_path=shared_config_path, override_config=dict(wandb.config))
    else:
        agent = AgentClass(env, agent_name, shared_config_path=shared_config_path, agent_config_path=agent_config_path)

    agent.train(effective_alpha)

    filename = str(f'run_names_{agent_type}.txt')
    with open(filename, 'a') as file:
        file.write(agent_name + '\n')

    print("Done Training with alpha: ", alpha, "agent_type: ", agent_type, "agent_name: ", agent_name)
    return agent_name

def run_sweep(env, shared_config_path, agent_type):
    shared_config = load_config(shared_config_path)
    run = wandb.init(project=shared_config['wandb']['project'], entity=shared_config['wandb']['entity'])
    config = run.config
    alpha = config.alpha
    print(alpha)
    agent_type = agent_type

    run_training(env, shared_config_path, alpha, agent_type, is_sweep=True)
    print("Running Sweep...")

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

def run_evaluation(env, shared_config_path, agent_type, alpha, run_name, csv_path=None):
    print("Running Evaluation...")
    print("csv_path: ", csv_path)

    # Load agent configuration
    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    # Initialize agent
    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name, shared_config_path=shared_config_path, agent_config_path=agent_config_path, csv_path=csv_path)

    # Run the evaluation
    avg_reward = agent.evaluate(run_name=run_name, alpha=alpha, csv_path=csv_path)
    print(f"Average Reward for {agent_type} agent: {avg_reward}")




def run_evaluation_random(env, shared_config_path, agent_type, alpha, run_name):
    print("Running Evaluation...")

    agent_config_path = os.path.join('config', f'config_{agent_type}.yaml')
    load_config(agent_config_path)

    AgentModule = __import__(f'{agent_type}.agent', fromlist=[f'{format_agent_class_name(agent_type)}'])
    AgentClass = getattr(AgentModule, f'{format_agent_class_name(agent_type)}')
    agent = AgentClass(env, run_name, shared_config_path=shared_config_path, agent_config_path=os.path.join('config', f'config_{agent_type}.yaml'))

    test_episodes = 4
    evaluation_metrics = agent.test_baseline_random(test_episodes, alpha)

    print("Evaluation Metrics for random agent:", evaluation_metrics)

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
    parser = argparse.ArgumentParser(description='Run training, evaluation, multiple runs, or a sweep.')
    parser.add_argument('mode', choices=['train', 'eval', 'random', 'sweep', 'multi', 'optuna'], help='Mode to run the script in.')
    parser.add_argument('--alpha', type=float, default=0.6, help='Reward parameter alpha.')
    parser.add_argument('--alpha_t', type=float, default=0.05, help='Alpha value for tolerance interval.')
    parser.add_argument('--beta_t', type=float, default=0.9, help='Beta value for tolerance interval.')
    parser.add_argument('--num_runs', type=int, default=20, help='Number of runs for tolerance interval.')
    parser.add_argument('--agent_type', default='q_learning', help='Type of agent to use.')
    parser.add_argument('--run_name', default=None, help='Unique name for the training run or evaluation.')
    parser.add_argument('--read_from_csv', action='store_true', help='Read community risk values from CSV.')
    parser.add_argument('--csv_path', default=None, help='Path to the CSV file containing community risk values.')

    global args
    args = parser.parse_args()

    shared_config_path = os.path.join('config', 'config_shared.yaml')
    env, shared_config = initialize_environment(shared_config_path, args.read_from_csv, args.csv_path)

    if args.mode == 'train':
        run_training(env, shared_config_path, args.alpha, args.agent_type)

    elif args.mode == 'eval':
        run_evaluation(env, shared_config_path, args.agent_type, args.alpha, args.run_name, args.csv_path)

    elif args.mode == 'random':
        run_evaluation_random(env, shared_config_path, args.agent_type, args.alpha, args.run_name)

    elif args.mode == 'sweep':
        sweep_config_path = os.path.join('config', 'sweep.yaml')
        sweep_config = load_config(sweep_config_path)
        sweep_id = wandb.sweep(sweep_config, project=shared_config['wandb']['project'],
                               entity=shared_config['wandb']['entity'])
        wandb.agent(sweep_id, function=lambda: run_sweep(env, shared_config_path, args.agent_type))

    elif args.mode == 'multi':
        run_multiple_runs(env, shared_config_path, args.agent_type, args.alpha_t, args.beta_t, args.num_runs)

    elif args.mode == 'optuna':
        run_optuna(env, shared_config_path, args.agent_type)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == '__main__':
    main()

