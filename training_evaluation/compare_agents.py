import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()


def downsample(data, factor):
    return data[::factor]


def plot_combined(qlearning_log, dqn_log, metric, title, ylabel, unit, window_size=100, downsample_factor=5):
    plt.figure(figsize=(12, 6))

    if metric not in qlearning_log.columns or metric not in dqn_log.columns:
        print(f"Metric '{metric}' not found in one or both datasets")
        return

    smoothed_qlearning = moving_average(qlearning_log[metric], window_size)
    smoothed_qlearning = downsample(smoothed_qlearning, downsample_factor)
    smoothed_dqn = moving_average(dqn_log[metric], window_size)
    smoothed_dqn = downsample(smoothed_dqn, downsample_factor)

    sns.lineplot(x=downsample(qlearning_log['episode'], downsample_factor), y=smoothed_qlearning, label='Q-Learning',
                 color='blue')
    sns.lineplot(x=downsample(dqn_log['episode'], downsample_factor), y=smoothed_dqn, label='DQN', color='orange')

    plt.title(title, fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(f'{ylabel} ({unit})', fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(
        f"{metric} - Q-Learning: min={smoothed_qlearning.min():.4f}, max={smoothed_qlearning.max():.4f}, mean={smoothed_qlearning.mean():.4f}")
    print(f"{metric} - DQN: min={smoothed_dqn.min():.4f}, max={smoothed_dqn.max():.4f}, mean={smoothed_dqn.mean():.4f}")


def main():
    # Load Q-Learning and DQN logs
    qlearning_log = pd.read_csv('training_metrics_q.csv')
    dqn_log = pd.read_csv('training_metrics_dqn.csv')

    # Plot metrics
    metrics = [
        ('cumulative_reward', 'Cumulative Reward', 'Cumulative Reward', 'points'),
        ('average_reward', 'Average Reward', 'Average Reward', 'points/episode'),
        ('discounted_reward', 'Discounted Reward', 'Discounted Reward', 'points'),
        ('convergence_rate', 'Convergence Rate', 'Convergence Rate', 'changes/episode'),
        ('sample_efficiency', 'Sample Efficiency', 'Sample Efficiency', 'steps/episode'),
        ('policy_consistency', 'Policy Consistency', 'Policy Consistency', 'ratio'),
        ('policy_entropy', 'Policy Entropy', 'Policy Entropy', 'nats'),
        ('time_complexity', 'Time Complexity', 'Time Complexity', 'seconds'),
        ('space_complexity', 'Space Complexity', 'Space Complexity', 'bytes')
    ]

    for metric, title, ylabel, unit in metrics:
        plot_combined(qlearning_log, dqn_log, metric, title, ylabel, unit)

    # Summary table
    summary_data = {
        'Metric': ['Time Complexity', 'Space Complexity', 'Steps'],
        'Q-Learning': [
            qlearning_log['time_complexity'].max(),
            qlearning_log['space_complexity'].max(),
            qlearning_log['step'].max()
        ],
        'DQN': [
            dqn_log['time_complexity'].max(),
            dqn_log['space_complexity'].max(),
            dqn_log['step'].max()
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)

    # Save summary table to CSV
    summary_df.to_csv('summary_metrics.csv', index=False)

    # Plot summary table
    plt.figure(figsize=(10, 2))
    sns.heatmap(summary_df.set_index('Metric').T, annot=True, cmap="YlGnBu", cbar=False, fmt='.2e')
    plt.title('Summary Metrics')
    plt.savefig('summary_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()