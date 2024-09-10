import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.patches as mpatches
from torch import nn
import torch.nn.functional as F
import csv
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D
import random
import colorsys

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Define allowed actions

def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.95) for x in range(n)]  # Adjusted the brightness for better distinction
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in RGB_tuples]
def calculate_allowed_values(N, levels):
    step = N // (levels - 1)
    allowed_values = torch.tensor([i * step for i in range(levels)])
    return allowed_values

# Example usage
N = 100
levels = 5
allowed = calculate_allowed_values(N, levels)  # This will give torch.tensor([0, 50, 100])

class LyapunovNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1):
        super(LyapunovNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

        # # Xavier initialization
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = F.softplus(self.fc3(out))  # Ensure positive output
        return out
class MyopicPolicy:
    def __init__(self):
        self.save_path = "myopic_exp_results"
        os.makedirs(self.save_path, exist_ok=True)

    def estimate_infected_students(self, current_infected, allowed_per_course, community_risk):
        alpha_m = 0.005
        beta = 0.01

        current_infected = current_infected.view(-1, 1, 1).to(dtype=torch.float32)
        community_risk = community_risk.view(-1, 1, 1).to(dtype=torch.float32)
        allowed_per_course = allowed_per_course.view(1, 1, -1).to(dtype=torch.float32)

        infected = ((alpha_m * current_infected) * allowed_per_course +
                    (beta * community_risk) * allowed_per_course ** 2)
        infected = torch.min(infected, allowed_per_course)
        return infected.squeeze(1).to(dtype=torch.float32)

    def get_reward(self, allowed, new_infected, alpha: float):
        reward = round((alpha * allowed) - ((1 - alpha) * new_infected))
        return reward

    def get_label(self, num_infected, community_risk, alpha):
        num_samples = num_infected.shape[0]
        num_actions = len(allowed)

        # Ensure input tensors are float32
        num_infected = num_infected.to(dtype=torch.float32)
        community_risk = community_risk.to(dtype=torch.float32)

        label = torch.zeros(num_samples, dtype=torch.long)
        max_reward = torch.full((num_samples,), -float('inf'), dtype=torch.float32)
        allowed_values = torch.zeros(num_samples, dtype=torch.float32)
        new_infected_values = torch.zeros(num_samples, dtype=torch.float32)
        reward_values = torch.zeros(num_samples, dtype=torch.float32)

        rewards_for_actions = torch.zeros((num_samples, num_actions), dtype=torch.float32)
        cumulative_rewards = torch.zeros(num_samples, dtype=torch.float32)

        allowed_per_course = allowed.view(1, -1).to(dtype=torch.float32)
        new_infected = self.estimate_infected_students(num_infected, allowed_per_course, community_risk)

        # Ensure new_infected is float32
        new_infected = new_infected.to(dtype=torch.float32)

        for i, a in enumerate(allowed):
            a = float(a)  # Ensure 'a' is a float
            reward = (alpha * a - (1 - alpha) * new_infected[:, i]).round().to(dtype=torch.float32)
            rewards_for_actions[:, i] = reward
            mask = reward > max_reward
            label[mask] = i
            max_reward[mask] = reward[mask]
            allowed_values[mask] = a
            new_infected_values[mask] = new_infected[:, i][mask].to(dtype=torch.float32)
            reward_values[mask] = reward[mask]
            cumulative_rewards += reward

        return label, allowed_values, new_infected_values, reward_values, rewards_for_actions, cumulative_rewards

    def identify_safety_set(self, allowed_values_over_time, infected_values_over_time, x_value, y_value, z):
        """Identify and plot the safety set based on fixed constraints."""
        infected_values_over_time = [val[0] if isinstance(val, (list, tuple)) else val for val in
                                     infected_values_over_time]

        # 1. Ensure that no more than `x_value` infected individuals are present for more than `z%` of the time.
        time_exceeding_x = sum(1 for val in infected_values_over_time if val > x_value)
        time_within_x = len(infected_values_over_time) - time_exceeding_x
        infection_safety_percentage = (time_within_x / len(infected_values_over_time)) * 100

        # 2. Ensure that `y_value` allowed students are present at least `z%` of the time.
        time_with_y_present = sum(1 for val in allowed_values_over_time if val >= y_value)
        attendance_safety_percentage = (time_with_y_present / len(allowed_values_over_time)) * 100

        with open(os.path.join(self.save_path, 'safety_set.txt'), 'w') as f:
            f.write(
                f"Safety Condition: No more than {x_value} infections for {100 - z}% of time: {infection_safety_percentage}%\n")
            f.write(
                f"Safety Condition: At least {y_value} allowed students for {z}% of time: {attendance_safety_percentage}%\n")

        plt.figure(figsize=(10, 6))
        plt.scatter(allowed_values_over_time, infected_values_over_time, color='blue', label='State Points')
        plt.axhline(y=x_value, color='red', linestyle='--', label=f'Infection Threshold (x={x_value})')
        plt.axvline(x=y_value, color='green', linestyle='--', label=f'Attendance Threshold (y={y_value})')
        plt.xlabel('Allowed Students')
        plt.ylabel('Infected Individuals')
        plt.legend()
        plt.title('Safety Set Identification')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'safety_set.png'))
        plt.close()

        infection_condition_met = infection_safety_percentage >= (100 - z)
        attendance_condition_met = attendance_safety_percentage >= z

        return infection_condition_met, attendance_condition_met

    def construct_cbf(self, allowed_values_over_time, infected_values_over_time, x_value, y_value):
        """Construct and save the Control Barrier Function (CBF) based on fixed safety constraints."""
        processed_infected_values = [
            infected[0] if isinstance(infected, (list, tuple)) else infected
            for infected in infected_values_over_time
        ]

        # Directly use the provided x_value and y_value
        B1 = lambda s: x_value - (s[1][0] if isinstance(s[1], (list, tuple)) else s[1])
        B2 = lambda s: (s[0] - y_value)

        with open(os.path.join(self.save_path, 'cbf.txt'), 'w') as f:
            f.write(f"CBF for Infections: B1(s) = {x_value} - Infected Individuals\n")
            f.write(f"CBF for Attendance: B2(s) = Allowed Students - {y_value}\n")

        return B1, B2

    def verify_forward_invariance(self, B1, B2, allowed_values_over_time, infected_values_over_time):
        """Verify forward invariance using the constructed CBFs."""
        is_invariant = True
        for i in range(len(allowed_values_over_time) - 1):
            s_t = [allowed_values_over_time[i], infected_values_over_time[i]]
            s_t_plus_1 = [allowed_values_over_time[i + 1], infected_values_over_time[i + 1]]

            dB1_dt = B1(s_t_plus_1) - B1(s_t)
            dB2_dt = B2(s_t_plus_1) - B2(s_t)

            if not (dB1_dt >= 0 and dB2_dt >= 0):
                is_invariant = False
                break

        with open(os.path.join(self.save_path, 'cbf_verification.txt'), 'w') as f:
            if is_invariant:
                f.write("The forward invariance of the system is verified using the constructed CBFs.\n")
            else:
                f.write("The system is not forward invariant based on the constructed CBFs.\n")

        return is_invariant

    def get_next_states(self, states, alpha):
        """
        Get the next states based on the current states and the policy.

        Args:
        states (torch.Tensor): Current states (infected, risk)
        alpha (float): Policy parameter

        Returns:
        torch.Tensor: Next states
        """
        next_states = []
        for state in states:
            infected, risk = state
            infected = infected.unsqueeze(0)
            risk = risk.unsqueeze(0)

            label, allowed_value, new_infected, _, _, _ = self.get_label(infected, risk, alpha)

            next_states.append(torch.tensor([new_infected.item(), risk.item()]))

        return torch.stack(next_states)

    def generate_diverse_states(self,num_samples):
        rng = np.random.default_rng(SEED)  # Create a new RNG with the seed
        infected = rng.uniform(0, 100, num_samples)
        risk = rng.uniform(0, 1, num_samples)
        return torch.tensor(np.column_stack((infected, risk)), dtype=torch.float32)

    def construct_lyapunov_function(self, features, alpha):
        model = LyapunovNet(input_dim=2, hidden_dim=64, output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        loss_values = []
        epochs = 1000
        epsilon = 1e-6

        train_states = self.generate_diverse_states(100)

        for epoch in range(epochs):
            optimizer.zero_grad()
            V = model(train_states)
            next_states = self.get_next_states(train_states, alpha)
            V_next = model(next_states)

            # Lyapunov conditions
            positive_definite_loss = F.relu(-V + 1).mean()
            decreasing_loss = F.relu(V_next - V + epsilon).mean()

            # Ensure V(0) = 0
            zero_state = torch.zeros_like(train_states[0]).unsqueeze(0)
            origin_loss = model(zero_state).squeeze() ** 2

            # Ensure V increases with distance from origin
            distance_from_origin = torch.norm(train_states, dim=1)
            distance_loss = F.relu(1 - V / (distance_from_origin + epsilon)).mean()

            loss = positive_definite_loss + decreasing_loss + origin_loss + distance_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}, "
                      f"PD Loss: {positive_definite_loss.item():.6f}, "
                      f"Dec Loss: {decreasing_loss.item():.6f}, "
                      f"Origin Loss: {origin_loss.item():.6f}, "
                      f"Distance Loss: {distance_loss.item():.6f}")

            loss_values.append(loss.item())

        torch.save(model.state_dict(), os.path.join(self.save_path, 'lyapunov_model.pth'))

        # Print final loss components and function values
        with torch.no_grad():
            V_final = model(train_states)
            V_next_final = model(next_states)
            print(
                f"Final V stats: min={V_final.min().item():.4f}, max={V_final.max().item():.4f}, mean={V_final.mean().item():.4f}")
            print(
                f"Final V_next stats: min={V_next_final.min().item():.4f}, max={V_next_final.max().item():.4f}, mean={V_next_final.mean().item():.4f}")

        return model, loss_values

    def evaluate_lyapunov(self, model, features, alpha):
        eval_states = torch.tensor([[f[1], f[2]] for f in features], dtype=torch.float32)

        with torch.no_grad():
            V = model(eval_states).squeeze()
            V_next = model(self.get_next_states(eval_states, alpha)).squeeze()

            positive_definite = (V > 0).float().mean()
            decreasing = (V_next < V).float().mean()

        print(f"Positive definite: {positive_definite.item():.2%}")
        print(f"Decreasing: {decreasing.item():.2%}")

        return positive_definite.item(), decreasing.item()

    def plot_loss_function(self, loss_values, alpha, run_name):
        if not loss_values:
            print("No loss values to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Lyapunov Function Training Loss (Run: {run_name}, Alpha: {alpha})')
        plt.yscale('log')  # Use log scale for y-axis
        plt.grid(True)
        plt.tight_layout()

        # Add text with min and max loss values
        min_loss = min(loss_values)
        max_loss = max(loss_values)
        plt.text(0.05, 0.95, f'Min Loss: {min_loss:.6f}\nMax Loss: {max_loss:.6f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.savefig(os.path.join(self.save_path, f'lyapunov_loss_plot_{run_name}_alpha_{alpha}.png'))
        plt.close()

        print(f"Min Loss: {min_loss}, Max Loss: {max_loss}")
    def verify_lyapunov_stability(self, V, features, alpha):
        positive_definite, decreasing = self.evaluate_lyapunov(V, features, alpha)

        with open(os.path.join(self.save_path, 'lyapunov_verification.txt'), 'w') as f:
            f.write(f"Lyapunov Function Evaluation:\n")
            f.write(f"Positive definite: {positive_definite:.2%}\n")
            f.write(f"Decreasing: {decreasing:.2%}\n")

            if positive_definite > 0.99 and decreasing > 0.99:
                f.write("The Lyapunov function satisfies stability conditions.\n")
            else:
                f.write("The Lyapunov function does not fully satisfy stability conditions.\n")

        return positive_definite > 0.99 and decreasing > 0.99

    def plot_steady_state_and_stable_points(self, model, features, run_name, alpha):
        infected = np.linspace(0, 100, 100)
        risk = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(infected, risk)
        states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        with torch.no_grad():
            V = model(states).squeeze().numpy().reshape(X.shape)

        # Compute steady-state distribution
        V_normalized = (V - V.min()) / (V.max() - V.min() + 1e-10)  # Normalize V to [0, 1]
        steady_state = np.exp(-V_normalized * 10)  # Scale factor to accentuate differences
        steady_state /= steady_state.sum()  # Normalize

        # Find local minima (asymptotically stable points)
        min_coords = argrelextrema(V, np.less, order=5)
        stable_points = list(zip(X[min_coords], Y[min_coords]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot steady-state distribution
        im1 = ax1.contourf(X, Y, steady_state, levels=20, cmap='viridis')
        ax1.set_title(f'Steady-State Distribution (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        fig.colorbar(im1, ax=ax1, label='Probability Density')

        # Plot Lyapunov function
        im2 = ax2.contourf(X, Y, V, levels=20, cmap='coolwarm')
        ax2.set_title(f'Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        fig.colorbar(im2, ax=ax2, label='V(x)')

        # Plot stable points
        for point in stable_points:
            ax2.plot(point[0], point[1], 'ko', markersize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'steady_state_and_stable_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

        # Print some statistics for debugging
        print(f"Lyapunov function stats: min={V.min():.4f}, max={V.max():.4f}, mean={V.mean():.4f}")
        print(
            f"Steady-state distribution stats: min={steady_state.min():.4e}, max={steady_state.max():.4e}, mean={steady_state.mean():.4e}")

    def plot_equilibrium_points(self, features, run_name, alpha):
        infected = [f[1] for f in features]
        risk = [f[2] for f in features]

        plt.figure(figsize=(10, 8))
        plt.scatter(risk, infected, c=infected, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Number of Infected')
        plt.xlabel('Community Risk')
        plt.ylabel('Number of Infected')
        plt.title(f'Equilibrium Points: DFE and EE Regions (Run: {run_name}, Alpha: {alpha})')

        # Define DFE and EE regions
        plt.axhline(y=0.5, color='r', linestyle='--', label='DFE/EE Boundary')
        plt.text(0.5, 0.25, 'DFE Region', horizontalalignment='center')
        plt.text(0.5, 0.75, 'EE Region', horizontalalignment='center')

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'equilibrium_points_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def plot_loss_function(self, loss_values, alpha, run_name):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Lyapunov Function Training Loss (Run: {run_name}, Alpha: {alpha})')
        plt.yscale('log')  # Use log scale for y-axis
        plt.grid(True)
        plt.tight_layout()

        # Add text with min and max loss values
        min_loss = min(loss_values)
        max_loss = max(loss_values)
        plt.text(0.05, 0.95, f'Min Loss: {min_loss:.6f}\nMax Loss: {max_loss:.6f}',
                 transform=plt.gca().transAxes, verticalalignment='top')

        plt.savefig(os.path.join(self.save_path, f'lyapunov_loss_plot_{run_name}_alpha_{alpha}.png'))
        plt.close()

        print(f"Min Loss: {min_loss}, Max Loss: {max_loss}")

    def visualize_policy(self, run_name, alpha):
        """Visualize the policy using the learned labels from get_label function by considering all possible combinations of infected and allowed values."""

        # Define the ranges for community risk and infected values
        community_risk_range = np.linspace(0, 1, 10)  # Community risk from 0 to 1 with 10 steps
        infected_range = np.arange(0, 100, 10)  # Infected individuals from 0 to 100 in steps of 10

        fig, ax = plt.subplots(figsize=(5, 5))

        # Generate distinct colors based on the number of allowed levels
        colors = generate_distinct_colors(len(allowed))
        color_map = {i: colors[i] for i in range(len(allowed))}

        for infected in infected_range:
            for community_risk in community_risk_range:
                current_infected_tensor = torch.tensor([infected], dtype=torch.float32)
                community_risk_tensor = torch.tensor([community_risk], dtype=torch.float32)

                label, allowed_value, _, _, _, _ = self.get_label(current_infected_tensor, community_risk_tensor, alpha)

                action = int(label.item())
                color = color_map[action]

                ax.scatter(community_risk, infected, color=color, s=100, marker='s')

        ax.set_xlabel('Community Risk')
        ax.set_ylabel('Infected Individuals')
        ax.set_title(f'Myopic')
        ax.grid(False)

        # Add padding to y-axis and x-axis
        # y_padding = 100 * 0.001  # 5% padding for the y-axis
        # ax.set_ylim(-y_padding, 100 + y_padding)
        ax.set_xlim(-0.05, 1.05)

        # Set y-ticks to show appropriate scale
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_yticklabels([f'{int(y)}' for y in ax.get_yticks()])

        # Create a custom legend
        legend_elements = [mpatches.Patch(facecolor=colors[i], label=f'{allowed[i]}%') for i in
                           range(len(allowed))]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 0.5), fontsize='large')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)

        # Save the plot with alpha in the filename
        plot_filename = os.path.join(self.save_path, f"policy_visualization_{run_name}_alpha_{levels}.png")
        print(f"Saving policy visualization plot to {plot_filename}")
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

    def get_next_state(self, current_state, alpha=0.2):
        current_infected = current_state[0].unsqueeze(0)
        community_risk = current_state[1].unsqueeze(0)

        label, allowed_value, new_infected, _, _, _ = self.get_label(current_infected, community_risk, alpha)

        return torch.tensor([new_infected.item(), community_risk.item()], dtype=torch.float32)

    def plot_lyapunov_change(self, model, features, run_name, alpha):
        infected = np.linspace(0, 100, 50)
        risk = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(infected, risk)
        states = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

        with torch.no_grad():
            V = model(states).squeeze().numpy().reshape(X.shape)

        # Calculate change in V
        delta_V = np.zeros_like(V)
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                current_state = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32)
                next_state = self.get_next_states(current_state.unsqueeze(0), alpha).squeeze()
                current_V = model(current_state.unsqueeze(0)).item()
                next_V = model(next_state.unsqueeze(0)).item()
                delta_V[i, j] = next_V - current_V

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Heatmap of V(x)
        im1 = ax1.imshow(V, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='viridis')
        ax1.set_title(f'Lyapunov Function V(x) (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('Infected')
        ax1.set_ylabel('Community Risk')
        fig.colorbar(im1, ax=ax1)

        # Heatmap of ΔV(x)
        im2 = ax2.imshow(delta_V, extent=[0, 100, 0, 1], origin='lower', aspect='auto', cmap='coolwarm')
        ax2.set_title(f'Change in Lyapunov Function ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('Infected')
        ax2.set_ylabel('Community Risk')
        fig.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'lyapunov_change_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def plot_lyapunov_properties(self, model, features, run_name, alpha):
        eval_states = torch.tensor([[f[1], f[2]] for f in features], dtype=torch.float32)

        with torch.no_grad():
            V = model(eval_states).squeeze()
            next_states = self.get_next_states(eval_states, alpha)
            V_next = model(next_states).squeeze()
            delta_V = V_next - V

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

        # Plot Lyapunov function values
        ax1.plot(V.numpy(), label='V(x)')
        ax1.axhline(y=0, color='r', linestyle='--', label='V(x) = 0')
        ax1.set_title(f'Lyapunov Function Values (Run: {run_name}, Alpha: {alpha})')
        ax1.set_xlabel('State Index')
        ax1.set_ylabel('V(x)')
        ax1.legend()
        ax1.grid(True)

        # Plot delta V
        ax2.plot(delta_V.numpy(), label='ΔV(x)')
        ax2.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax2.set_title(f'Change in Lyapunov Function (Run: {run_name}, Alpha: {alpha})')
        ax2.set_xlabel('State Index')
        ax2.set_ylabel('ΔV(x)')
        ax2.legend()
        ax2.grid(True)

        # Plot V(x) vs ΔV(x)
        ax3.scatter(V.numpy(), delta_V.numpy(), alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--', label='ΔV(x) = 0')
        ax3.axvline(x=0, color='g', linestyle='--', label='V(x) = 0')
        ax3.set_title(f'V(x) vs ΔV(x) (Run: {run_name}, Alpha: {alpha})')
        ax3.set_xlabel('V(x)')
        ax3.set_ylabel('ΔV(x)')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'lyapunov_properties_{run_name}_alpha_{alpha}.png'))
        plt.close()

        positive_definite = (V > 0).float().mean()
        decreasing = (delta_V < 0).float().mean()

        print(f"Positive definite: {positive_definite.item():.2%}")
        print(f"Decreasing: {decreasing.item():.2%}")

        return positive_definite.item(), decreasing.item()

    def plot_safety_analysis_results(self, xy_values, z_values, infection_safety_results, attendance_safety_results,
                                     evaluation_subdirectory, run_name, alpha):
        plt.figure(figsize=(14, 8))

        # Heatmap for infection safety results
        plt.subplot(1, 2, 1)
        sns.heatmap(infection_safety_results, annot=True, cmap='RdBu', xticklabels=z_values,
                    yticklabels=list(reversed(xy_values)), cbar=False, vmin=0, vmax=1)
        plt.xlabel('z Values (Threshold Percentage)')
        plt.ylabel('x Values (Infection/Attendance Threshold)')
        plt.title(f'Infection Safety Analysis Heatmap (Run: {run_name}, Alpha: {alpha})')
        plt.text(0.5, -0.1, "0: Safety Condition Not Met, 1: Safety Condition Met", ha='center', va='center',
                 transform=plt.gca().transAxes)

        # Heatmap for attendance safety results
        plt.subplot(1, 2, 2)
        sns.heatmap(attendance_safety_results, annot=True, cmap='RdBu', xticklabels=z_values,
                    yticklabels=list(reversed(xy_values)), cbar=False, vmin=0, vmax=1)
        plt.xlabel('z Values (Threshold Percentage)')
        plt.ylabel('x Values (Infection/Attendance Threshold)')
        plt.title(f'Attendance Safety Analysis Heatmap (Run: {run_name}, Alpha: {alpha})')
        plt.text(0.5, -0.1, "0: Safety Condition Not Met, 1: Safety Condition Met", ha='center', va='center',
                 transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(evaluation_subdirectory, f'safety_analysis_heatmap_{run_name}_alpha_{alpha}.png'))
        plt.close()
    def identify_safety_set_for_values(self, x, y, z):
        """Identify safety conditions for given x, y, and z values."""
        # This should replicate the logic in your identify_safety_set method
        # Replace with the actual computation for specific x, y, z values
        # For demonstration purposes:
        infection_condition_met = x < 25  # Example condition for infections
        attendance_condition_met = y > 50  # Example condition for attendance
        return infection_condition_met, attendance_condition_met

    def safety_analysis(self, x_values, y_values, z_values, run_name, num_episodes=1, alpha=0.5, csv_path=None):
        """Performs a safety analysis as a function of x, y, and z values, and generates plots."""
        evaluation_subdirectory = os.path.join(self.save_path, run_name)
        os.makedirs(evaluation_subdirectory, exist_ok=True)

        if csv_path:
            community_risk_values = pd.read_csv(csv_path)['Risk-Level'].values
            num_weeks = len(community_risk_values)
        else:
            num_weeks = 52
            community_risk_values = np.random.uniform(0, 1, num_weeks)

        total_rewards = []

        # Arrays to store results
        infection_safety_results = np.zeros((len(x_values), len(z_values)))
        attendance_safety_results = np.zeros((len(y_values), len(z_values)))

        for i, x in enumerate(x_values):
            for j, z in enumerate(z_values):
                for k, y in enumerate(y_values):

                    allowed_values_over_time = []
                    infected_values_over_time = []

                    for episode in range(num_episodes):
                        current_infected = torch.tensor([20.0], dtype=torch.float32)
                        total_reward = 0

                        for step in range(num_weeks):
                            community_risk = torch.tensor([community_risk_values[step]], dtype=torch.float32)

                            label, allowed_value, new_infected, reward, _, _ = self.get_label(current_infected,
                                                                                              community_risk,
                                                                                              alpha)

                            total_reward += reward.item()

                            if episode == 0:
                                allowed_values_over_time.append(allowed_value.item())
                                infected_values_over_time.append(new_infected.item())

                            current_infected = new_infected

                        total_rewards.append(total_reward)
                        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

                    # # Perform safety analysis for the current x, y, and z
                    # infection_condition_met, attendance_condition_met = self.identify_safety_set(
                    #     allowed_values_over_time,
                    #     infected_values_over_time,
                    #     x, y, z
                    # )
                    #
                    # # Store the results
                    # infection_safety_results[i, j] = infection_condition_met
                    # attendance_safety_results[k, j] = attendance_condition_met

        # Plot the results
        # self.plot_safety_analysis_results(x_values, z_values, infection_safety_results,
        #                                   attendance_safety_results, evaluation_subdirectory, run_name, alpha)

    def plot_safety_function(self, x_range, y_range, alpha_values, run_name):
        """Plot the safety function heatmap for different alpha values with contour lines and larger fonts."""
        plt.rcParams.update({'font.size': 14})  # Increase the default font size

        for alpha in alpha_values:
            safety_function = np.zeros((len(x_range), len(y_range)))
            for i, x in enumerate(x_range):
                for j, y in enumerate(y_range):
                    safety_function[i, j] = (alpha * y) - ((1 - alpha) * x)

            plt.figure(figsize=(12, 10))  # Slightly larger figure size
            heatmap = sns.heatmap(safety_function, cmap="coolwarm", cbar=True,
                                  xticklabels=np.round(y_range, 1),
                                  yticklabels=np.round(x_range, 1))

            plt.gca().invert_yaxis()


            plt.xlabel('y (Allowed Students Threshold)', fontsize=18)
            plt.ylabel('x (Infection Threshold)', fontsize=18)
            plt.title(f'(gamma={alpha})', fontsize=24)

            colorbar = heatmap.collections[0].colorbar
            colorbar.set_label('Safety Function Value', fontsize=18)

            # Increase font size for tick labels
            plt.tick_params(axis='both', which='major', labelsize=14)

            plt.tight_layout()  # Adjust the layout to prevent cutting off labels
            plt.savefig(os.path.join(self.save_path, f'safety_function_heatmap_alpha_{alpha}.png'), dpi=300)
            plt.close()

        plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})  # Reset to default font size

    def plot_transition_matrix_using_risk(self, states, next_states, community_risks, run_name, alpha_value):
        """Plot and save the transition probability matrix using the extracted community risk values."""

        # Convert numpy arrays to tuples to make them hashable
        states = [tuple(state) for state in states]
        next_states = [tuple(next_state) for next_state in next_states]

        # Get unique states and map them to indices
        unique_states = sorted(set(states + next_states))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}

        num_states = len(unique_states)

        # Initialize the transition matrix
        P = np.zeros((num_states, num_states))

        # Fill the transition matrix with community risk considerations
        for state, next_state, risk in zip(states, next_states, community_risks):
            transition_probability = risk  # Use community risk to weight transitions
            P[state_to_idx[state], state_to_idx[next_state]] += transition_probability

        # Normalize each row to get probabilities
        P = P / P.sum(axis=1, keepdims=True)

        # Apply log scale for better visualization of low probabilities
        P_log = np.log10(P + 1e-10)

        # Plot the transition matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(P_log, cmap='gray', aspect='auto', origin='lower')
        plt.colorbar(label='Log10 of Transition Probability')
        plt.xlabel('Next State Index')
        plt.ylabel('Current State Index')
        plt.title(f'Transition Probability Matrix (Log Scale)\nRun: {run_name}, Alpha: {alpha_value}')
        plt.tight_layout()

        # Save the plot with alpha in the filename
        plot_filename = os.path.join(self.save_path, f'transition_matrix_{run_name}_{levels}.png')
        print(f"Saving transition matrix plot to {plot_filename}")
        plt.savefig(plot_filename)
        plt.close()

    import csv
    import os
    import matplotlib.pyplot as plt

    def find_optimal_xy(self, infected_values, allowed_values, community_risk_values, z=95, run_name=None,
                        evaluation_subdirectory=None, alpha=0.5):
        # Binary search for smallest x (infection threshold)
        low_x, high_x = 0, max(infected_values)
        optimal_x = high_x

        while low_x <= high_x:
            mid_x = (low_x + high_x) // 2
            time_lessthan_mid_x = sum(1 for val in infected_values if val < mid_x)
            infection_safety_percentage = ((len(infected_values) - time_lessthan_mid_x) / len(infected_values)) * 100

            if infection_safety_percentage >= z:
                optimal_x = mid_x
                high_x = mid_x - 1
            else:
                low_x = mid_x + 1

        # Binary search for largest y (allowed threshold)
        low_y, high_y = 0, max(allowed_values)
        optimal_y = low_y

        while low_y <= high_y:
            mid_y = (low_y + high_y) // 2
            time_with_mid_y_present = sum(1 for val in allowed_values if val >= mid_y)
            attendance_safety_percentage = (time_with_mid_y_present / len(allowed_values)) * 100

            if attendance_safety_percentage >= z:
                optimal_y = mid_y
                low_y = mid_y + 1
            else:
                high_y = mid_y - 1

        # Calculate the final safety percentages
        time_above_optimal_x = sum(1 for val in infected_values if val > optimal_x)
        final_infection_safety_percentage = (time_above_optimal_x / len(infected_values)) * 100

        time_with_optimal_y_present = sum(1 for val in allowed_values if val >= optimal_y)
        final_attendance_safety_percentage = (time_with_optimal_y_present / len(allowed_values)) * 100

        # Determine if the safety conditions are met
        infection_condition_met = final_infection_safety_percentage <= (100 - z)
        attendance_condition_met = final_attendance_safety_percentage >= z

        # Save the safety condition results
        if run_name and evaluation_subdirectory:
            output_file_path = os.path.join(evaluation_subdirectory, f'safety_conditions_{run_name}_{levels}.csv')

            with open(output_file_path, mode='w', newline='') as safety_file:
                safety_writer = csv.writer(safety_file)
                # Write the header
                safety_writer.writerow(['Run Name', 'Optimal X', 'Infection Safety %', 'Infection Condition Met',
                                        'Optimal Y', 'Attendance Safety %', 'Attendance Condition Met'])

                # Write the results
                safety_writer.writerow([
                    run_name,
                    optimal_x,
                    final_infection_safety_percentage,
                    'Yes' if infection_condition_met else 'No',
                    optimal_y,
                    final_attendance_safety_percentage,
                    'Yes' if attendance_condition_met else 'No'
                ])

        # Plotting the safety conditions
        if run_name and evaluation_subdirectory:
            plt.figure(figsize=(10, 6))
            plt.scatter(allowed_values, infected_values, color='blue', label='State Points')
            plt.axhline(y=optimal_x, color='red', linestyle='--', label=f'Infection Threshold (x={int(optimal_x)})')
            plt.axvline(x=optimal_y, color='green', linestyle='--', label=f'Attendance Threshold (y={int(optimal_y)})')

            plt.xlabel('Allowed Students')
            plt.ylabel('Infected Individuals')
            plt.legend()
            plt.title(f'Safety Set Identification - {run_name}')
            plt.grid(True)
            plt.savefig(os.path.join(evaluation_subdirectory, f'safety_set_plot_3_{run_name}_{levels}.png'))
            plt.close()

        return optimal_x, optimal_y

    def plot_evaluation_results(self, infected_values, allowed_values, community_risk_values, run_name, alpha):
        # Ensure all arrays have the same length
        min_length = min(len(infected_values), len(allowed_values), len(community_risk_values))
        infected_values = infected_values[:min_length]
        allowed_values = allowed_values[:min_length]
        community_risk_values = community_risk_values[:min_length]

        x = np.arange(len(infected_values))

        fig, ax1 = plt.subplots(figsize=(12, 8))
        sns.set(style="whitegrid")

        # Bar plot for infected and allowed
        ax1.bar(x - 0.2, infected_values, width=0.4, color='red', label='Infected', alpha=0.6)
        ax1.bar(x + 0.2, allowed_values, width=0.4, color='blue', label='Allowed', alpha=0.6)


        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Infected / Allowed Values')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # Create a secondary y-axis for community risk
        ax2 = ax1.twinx()
        # ax2.plot(x, community_risk_values, color='black', marker='o', label='Community Risk')
        sns.lineplot(x=x, y=community_risk_values, marker='s', linestyle='--', color='black', linewidth=2.5, ax=ax2)
        ax2.set_ylabel('Community Risk')
        ax2.legend(loc='upper right')

        plt.title(f'Evaluation Results\nRun: {run_name}')

        # Save the plot
        plot_filename = os.path.join(self.save_path, f'evaluation_results_{run_name}_{levels}.png')
        plt.savefig(plot_filename)
        plt.close()
        # Calculate the mean of allowed and infected values
        mean_allowed = sum(allowed_values) / len(allowed_values)
        mean_infected = sum(infected_values) / len(infected_values)

        # Save the means to a CSV file
        csv_file_path = os.path.join(self.save_path, f"evaluation_summary_{run_name}_{levels}.csv")
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Mean Allowed', 'Mean Infected'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'Mean Allowed': mean_allowed, 'Mean Infected': mean_infected})
        print(f"Saved evaluation plot to {plot_filename}")


    def simulate_steady_state(self, num_simulations, num_steps, alpha):
        """
        Simulate the system for a long time to estimate the steady-state distribution.

        Args:
        num_simulations (int): Number of independent simulations to run.
        num_steps (int): Number of steps for each simulation.
        alpha (float): The alpha parameter for the policy.

        Returns:
        numpy.ndarray: Array of final states from all simulations.
        """
        final_states = []

        for _ in range(num_simulations):
            infected = np.random.uniform(0, 100)  # Initial infected count
            risk = np.random.uniform(0, 1)  # Initial community risk

            for _ in range(num_steps):
                infected_tensor = torch.tensor([infected], dtype=torch.float32)
                risk_tensor = torch.tensor([risk], dtype=torch.float32)

                _, allowed_value, new_infected, _, _, _ = self.get_label(infected_tensor, risk_tensor, alpha)

                infected = new_infected.item()
                risk = np.random.uniform(0, 1)  # Assuming risk is randomly distributed each step

            final_states.append([infected, risk])

        return np.array(final_states)

    def plot_simulated_steady_state(self, final_states, run_name, alpha):
        """
        Plot the simulated steady-state distribution.

        Args:
        final_states (numpy.ndarray): Array of final states from simulations.
        run_name (str): Name of the current run for saving the plot.
        alpha (float): The alpha parameter used in the simulation.
        """
        plt.figure(figsize=(10, 8))

        # Calculate the 2D histogram
        hist, xedges, yedges = np.histogram2d(final_states[:, 1], final_states[:, 0], bins=50,
                                              range=[[0, 1], [0, 100]])

        # Plot the 2D histogram
        plt.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='gray_r')

        plt.colorbar(label='Frequency')
        plt.xlabel('Community Risk')
        plt.ylabel('Infected Individuals')
        plt.title(f'Simulated Steady-State Distribution\nRun: {run_name}, Alpha: {alpha}')

        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'simulated_steady_state_{run_name}_alpha_{alpha}.png'))
        plt.close()

    def save_mean_allowed_infected(self, allowed_means_per_episode, infected_means_per_episode, alpha, run_name, levels):
        # Flatten the list of lists before computing the sum
        allowed_means_flat = [item for sublist in allowed_means_per_episode for item in sublist]
        infected_means_flat = [item for sublist in infected_means_per_episode for item in sublist]

        # Compute the mean values
        mean_allowed = round(sum(allowed_means_flat) / len(allowed_means_flat))
        mean_infected = round(sum(infected_means_flat) / len(infected_means_flat))

        # Save the results in a separate CSV file
        summary_file_path = os.path.join(self.save_path, f'summary_{run_name}_{alpha}_{levels}.csv')
        with open(summary_file_path, 'w', newline='') as summary_csvfile:
            summary_writer = csv.DictWriter(summary_csvfile, fieldnames=['mean_allowed', 'mean_infected'])
            summary_writer.writeheader()
            summary_writer.writerow({'mean_allowed': mean_allowed, 'mean_infected': mean_infected})

    def evaluate(self, run_name, num_episodes=1, alpha=0.5, csv_path=None, x_value=20, y_value=50, z=95,
                 perform_safety_analysis=True, num_simulations=1000, num_steps=1000):
        if csv_path:
            community_risk_values = pd.read_csv(csv_path)['Risk-Level'].values
            num_weeks = len(community_risk_values)
        else:
            num_weeks = 52
            rng = np.random.default_rng(SEED)  # Create a new RNG with the seed
            community_risk_values = rng.uniform(0, 1, num_weeks)

        total_rewards = []
        allowed_values_over_time = []
        infected_values_over_time = [20]  # Assuming an initial infection count

        csv_file_path = os.path.join(self.save_path, f'evaluation_metrics_{run_name}.csv')
        safety_log_path = os.path.join(self.save_path, f'safety_conditions_{run_name}.csv')
        interpretation_path = os.path.join(self.save_path, f'safety_conditions_interpretation_{run_name}_{levels}.txt')

        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Step', 'State', 'Action', 'Community Risk', 'Total Reward'])

            states = []
            next_states = []
            community_risks = []

            for episode in range(num_episodes):
                current_infected = torch.tensor([20.0], dtype=torch.float32)
                total_reward = 0

                for step in range(num_weeks):
                    community_risk = torch.tensor([community_risk_values[step]], dtype=torch.float32)

                    label, allowed_value, new_infected, reward, _, _ = self.get_label(current_infected,
                                                                                      community_risk, alpha)

                    # Store the current state, next state, and community risk
                    states.append(current_infected.numpy())
                    next_states.append(new_infected.numpy())
                    community_risks.append(community_risk.item())

                    total_reward += reward.item()

                    writer.writerow(
                        [episode + 1, step + 1, current_infected.item(), allowed_value.item(),
                         community_risk.item(), reward.item()]
                    )

                    if episode == 0:
                        allowed_values_over_time.append(allowed_value.item())
                        infected_values_over_time.append(new_infected.item())

                    current_infected = new_infected

                total_rewards.append(total_reward)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

                # Ensure all arrays have the same length
                min_length = min(len(allowed_values_over_time), len(infected_values_over_time),
                                 len(community_risk_values))
                allowed_values_over_time = allowed_values_over_time[:min_length]
                infected_values_over_time = infected_values_over_time[:min_length]
                community_risk_values = community_risk_values[:min_length]

                # Identify the safety set
                is_safe_infection, is_safe_attendance = self.identify_safety_set(allowed_values_over_time,
                                                                                 infected_values_over_time, x_value,
                                                                                 y_value, z)

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

        # Construct CBFs
        B1, B2 = self.construct_cbf(allowed_values_over_time, infected_values_over_time, x_value, y_value)

        # Verify forward invariance
        is_invariant = self.verify_forward_invariance(B1, B2, allowed_values_over_time, infected_values_over_time)

        # Construct Lyapunov function
        # features = list(zip(allowed_values_over_time, infected_values_over_time, community_risk_values))
        # V, loss_values = self.construct_lyapunov_function(features, alpha)
        #
        # # Plot the Lyapunov loss function
        # self.plot_loss_function(loss_values, alpha, run_name)
        #
        # # Verify Lyapunov stability
        # self.verify_lyapunov_stability(V, features, alpha)
        #
        # # Plot Lyapunov-related graphs
        # self.plot_steady_state_and_stable_points(V, features, run_name, alpha)
        # self.plot_lyapunov_change(V, features, run_name, alpha)
        # self.plot_lyapunov_properties(V, features, run_name, alpha)
        # self.plot_equilibrium_points(features, run_name, alpha)


        # Additional plot for visualizing policy
        self.visualize_policy(run_name, alpha)
        self.plot_transition_matrix_using_risk(states, next_states, community_risks, run_name, alpha)

        # Perform safety analysis over ranges if specified
        if perform_safety_analysis:
            x_values = np.arange(10, 31, 5)
            y_values = np.arange(10, 31, 5)
            z_values = np.arange(80, 100, 5)
            self.safety_analysis(x_values, y_values, z_values, run_name, num_episodes, alpha, csv_path)

        x_values = np.linspace(1, 100, 10)
        y_values = np.linspace(0, 100, 10)
        z_value = 95
        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.plot_safety_function(x_values, y_values, alpha_values, run_name)
        # After the evaluation loop
        self.plot_evaluation_results(infected_values_over_time, allowed_values_over_time, community_risk_values,
                                     run_name, alpha)

        # Call find_optimal_xy after the evaluation
        optimal_x, optimal_y = self.find_optimal_xy(
            infected_values_over_time,
            allowed_values_over_time,
            community_risk_values,
            z=z,
            run_name=run_name,
            evaluation_subdirectory=self.save_path,
            alpha=alpha
        )

        print(f"Optimal x: {optimal_x}, Optimal y: {optimal_y}")
        allowed_means_per_episode = [allowed_values_over_time]  # List to store mean allowed values
        infected_means_per_episode = [infected_values_over_time]  # List to store mean infected values

        # Calculate and save the means
        self.save_mean_allowed_infected(allowed_means_per_episode, infected_means_per_episode, alpha, run_name,levels)

        # After the existing steady-state plot
        # final_states = self.simulate_steady_state(num_simulations, num_steps, alpha)
        # self.plot_simulated_steady_state(final_states, run_name, alpha)


        return avg_reward


def main(seed=42):
    global SEED
    SEED = seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    myopic_policy = MyopicPolicy()

    # Define parameters
    run_name = "myopic_test-3"
    num_episodes = 1
    csv_path = "aggregated_weekly_risk_levels.csv"
    x_value = 20  # 50% of average infected
    y_value = 50  # 50% of average allowed
    z = 95
    perform_safety_analysis = True
    num_simulations = 10000  # New parameter for steady-state simulation
    num_steps = 100  # New parameter for steady-state simulation

    # List of alpha values to iterate over
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # alpha_values = [0.1]

    for alpha in alpha_values:
        print(f"Running evaluation for alpha = {alpha}")
        current_run_name = f"{run_name}_alpha_{alpha}"

        avg_reward = myopic_policy.evaluate(
            run_name=current_run_name,
            num_episodes=num_episodes,
            alpha=alpha,
            csv_path=csv_path,
            x_value=x_value,
            y_value=y_value,
            z=z,
            perform_safety_analysis=perform_safety_analysis,
            num_simulations=num_simulations,
            num_steps=num_steps
        )

        print(f"Average Reward for alpha {alpha}: {avg_reward}")

if __name__ == "__main__":
    main()
