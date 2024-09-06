import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg
import random
from matplotlib.animation import FuncAnimation
import os

class StochasticStabilitySimulation:
    def __init__(self, N, alpha_m, beta, community_risk_method='uniform', community_risk_file=None, seed=None, max_weeks=52):
        self.N = N
        self.max_infected = N
        self.max_allowed = N
        self.alpha_m = alpha_m
        self.beta = beta
        self.community_risk_method = community_risk_method
        self.max_weeks = max_weeks
        self.community_risk = None
        self.episode_seed = seed if seed is not None else 42
        self.alpha = 0.2

        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility
            random.seed(seed)

        # Load community risk data if the method is 'csv'
        if community_risk_method == 'data' and community_risk_file:
            self.load_community_risk_data(community_risk_file)
        elif community_risk_method == 'sinusoidal':
            self.risk_values = self.generate_episode_risk()
            self.risk_iterator = iter(self.risk_values)
            self.community_risk = next(self.risk_iterator)

    def load_community_risk_data(self, file_path):
        data = pd.read_csv(file_path)
        self.community_risk = data['Risk-Level'].mean()  # Calculate the mean risk value

    def generate_episode_risk(self):
        """Generate risk values for a single episode."""
        self.episode_seed += 1
        random.seed(self.episode_seed)
        np.random.seed(self.episode_seed)

        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        num_components = random.randint(1, 3)  # Use 1 to 3 sine components
        risk_pattern = np.zeros(self.max_weeks)

        for _ in range(num_components):
            amplitude = random.uniform(0.2, 0.4)
            frequency = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2 * np.pi)
            risk_pattern += amplitude * np.sin(frequency * t + phase)

        risk_pattern = (risk_pattern - np.min(risk_pattern)) / (np.max(risk_pattern) - np.min(risk_pattern))
        risk_pattern = 0.9 * risk_pattern + 0.0  # Scale to range [0.1, 0.9]

        return [max(0.0, min(1.0, risk + random.uniform(-0.1, 0.1))) for risk in risk_pattern]

    def calculate_R0(self, allowed_values):
        # Ensure allowed_values is a numpy array
        allowed_values = np.asarray(allowed_values, dtype=float)

        if self.community_risk_method == 'uniform':
            community_risk = np.mean(np.random.uniform(0, 1, 1000))
        elif self.community_risk_method == 'data':
            if self.community_risk is None:
                raise ValueError("Community risk data not loaded properly from CSV.")
            community_risk = self.community_risk  # This should be a scalar value (mean of the CSV data)
        elif self.community_risk_method == 'sinusoidal':
            community_risk = self.generate_episode_risk()
            # If community_risk is a list, aggregate it (e.g., take the mean)
            if isinstance(community_risk, list):
                community_risk = np.mean(community_risk)
        else:
            raise ValueError(f"Unknown community risk method: {self.community_risk_method}")

        # Calculate R0 using element-wise operations on the numpy array
        R0_values = self.alpha_m * allowed_values + self.beta * community_risk * allowed_values ** 2
        return R0_values

    def create_transition_matrix(self, risk_func, num_simulations=1000):
        states = [(100, i) for i in range(self.max_infected + 1)]
        num_states = len(states)
        P = np.zeros((num_states, num_states))

        community_risks = np.random.uniform(0, 1, (num_simulations, num_states))

        for idx, (a, i) in enumerate(states):
            new_infections = np.minimum(
                (self.alpha_m * i * a + self.beta * community_risks[:, idx] * a ** 2).astype(int),
                self.max_infected
            )

            next_i = new_infections

            next_indices = np.array([
                states.index((100, next_i[j])) for j in range(num_simulations)
            ])

            transition_counts = np.bincount(next_indices, minlength=num_states)

            P[idx, :] = transition_counts / num_simulations

        return P, states

    def calculate_stationary_distribution(self, P):
        eigenvalues, eigenvectors = linalg.eig(P.T)
        stationary_index = np.argmin(np.abs(eigenvalues - 1))
        stationary = eigenvectors[:, stationary_index].real
        stationary /= stationary.sum()  # Normalize the distribution
        return stationary

    def check_ergodicity(self, P):
        num_states = P.shape[0]
        reachability_matrix = np.linalg.matrix_power(P, num_states)
        is_irreducible = np.all(reachability_matrix > 0)
        is_aperiodic = np.all(np.diag(P) > 0)
        return is_irreducible and is_aperiodic

    def analyze_stochastic_stability(self, stationary, states):
        infected_indices = [i for i, (_, inf) in enumerate(states) if inf > 0]
        dfe_probability = sum(stationary[i] for i, (_, inf) in enumerate(states) if inf == 0)
        ee_probability = sum(stationary[i] for i in infected_indices)

        print(f"Probability of Disease-Free Equilibrium (DFE) being stochastically stable: {dfe_probability:.4f}")
        print(f"Probability of Endemic Equilibrium (EE) being stochastically stable: {ee_probability:.4f}")

        if dfe_probability > ee_probability:
            print("The Disease-Free Equilibrium (DFE) is more stochastically stable.")
        else:
            print("The Endemic Equilibrium (EE) is more stochastically stable.")

        return dfe_probability, ee_probability

    def plot_stationary_distribution(self, stationary, states, method_label):
        x_values_fixed = []
        y_values_fixed = []
        probabilities_fixed = []

        for (a, i), prob in zip(states, stationary):
            community_risk = i / a if a > 0 else 0
            if a == self.max_allowed:
                x_values_fixed.append(community_risk)
                y_values_fixed.append(i)
                probabilities_fixed.append(prob)

        plt.figure(figsize=(10, 8))
        scatter_fixed = plt.scatter(x_values_fixed, y_values_fixed, c=probabilities_fixed,
                                    cmap='viridis', s=50, edgecolor='black')
        plt.colorbar(scatter_fixed, label='Probability')
        plt.xlabel('Community Risk')
        plt.ylabel('Number of Infected')
        plt.title(f'Stationary Distribution (Fixed Allowed Value)\nMethod: {method_label}')
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(f'stoch_plots/stationary_distribution_{method_label}.png')
        # plt.show()

    def plot_transition_matrix_for_policy(self, P, method_label, alpha_value, run_name):
        plt.figure(figsize=(10, 8))
        plt.imshow(P, cmap='gray', aspect='auto', origin='lower', vmin=0, vmax=0.01)
        plt.colorbar(label='Transition Probability')
        plt.xlabel('Next State Index')
        plt.ylabel('Current State Index')
        plt.title(f'Transition Probability Matrix\nMethod: {method_label}, Alpha: {alpha_value}')
        plt.tight_layout()
        plt.savefig(f'stoch_plots/transition_matrix_{run_name}_alpha_{alpha_value}.png')
        # plt.show()

    def plot_lyapunov_function_with_context(self, P, states, stationary, method_label):
        V = np.array([0.5 * (i ** 2 + a ** 2) for (a, i) in states])
        expected_V_next = P @ V
        delta_V = expected_V_next - V

        dfe_indices = [i for i, (_, inf) in enumerate(states) if inf == 0]
        ee_indices = [i for i, (_, inf) in enumerate(states) if inf > 0]

        plt.figure(figsize=(10, 6))

        plt.scatter(V[dfe_indices], delta_V[dfe_indices], color='blue', s=5, label='DFE region (Infected = 0)')
        plt.scatter(V[ee_indices], delta_V[ee_indices], color='red', s=5, label='EE region (Infected > 0)')
        plt.axhline(y=0, color='black', linestyle='--', label=r'$\Delta V = 0$')

        plt.xlabel('Lyapunov Function Value $V$')
        plt.ylabel(r'$\Delta V$')
        plt.title(f'Lyapunov Function Behavior in DFE and EE Regions\nMethod: {method_label}')
        plt.legend()
        plt.grid(True)
        plt.xlim(left=np.min(V[dfe_indices]) - 100, right=np.max(V) + 100)
        plt.ylim(bottom=np.min(delta_V) - 100, top=np.max(delta_V) + 100)
        plt.tight_layout()
        plt.savefig(f'stoch_plots/lyapunov_function_with_context_{method_label}.png')
        plt.savefig(f'stoch_plots/lyapunov_function_with_context_{method_label}.png')
        # plt.show()

    def plot_threshold_behavior_alpha_beta(self, method_label):
        alpha_values = np.linspace(0.0001, 0.01, 100)
        beta_values = np.linspace(0.0001, 0.01, 100)

        alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)

        community_risk = np.random.uniform(0, 1, alpha_grid.shape)

        infected_grid = np.clip(self.N * (alpha_grid + beta_grid * community_risk * self.N), 0, self.N)

        plt.figure(figsize=(10, 8))

        im = plt.contourf(alpha_grid, beta_grid, infected_grid, cmap='coolwarm', levels=100)
        plt.colorbar(im, label='Number of Infected')

        R0_contour = plt.contour(alpha_grid, beta_grid, infected_grid, levels=[1], colors='black', linestyles='--')
        if len(R0_contour.allsegs[0]) > 0:
            plt.clabel(R0_contour, inline=1, fontsize=10, fmt=lambda x: f'R0={x:.1f}')

        plt.xlabel('Alpha')
        plt.ylabel('Beta')
        plt.title(f'Threshold Behavior of R0 for Different Alpha and Beta\nMethod: {method_label}')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(f'stoch_plots/threshold_behavior_alpha_beta_{method_label}.png')
        # plt.show()

    def plot_transition_matrix(self, P, method_label):
        # Avoid log(0) by setting a small floor value for the log scale
        P_log = np.log10(P + 1e-10)  # Adding a small epsilon to avoid log(0)

        plt.figure(figsize=(10, 8))
        plt.imshow(P_log, cmap='gray', aspect='auto', origin='lower')
        plt.colorbar(label='Log10 of Transition Probability')
        plt.xlabel('Next State Index')
        plt.ylabel('Current State Index')
        plt.title(f'Transition Probability Matrix (Log Scale)\nMethod: {method_label}')
        plt.xlim(left=0, right=P.shape[1] - 1)
        plt.ylim(bottom=0, top=P.shape[0] - 1)
        plt.tight_layout()
        plt.savefig(f'stoch_plots/transition_matrix_{method_label}.png')
        plt.close()

    def plot_threshold_behavior_R0(self, method_label):
        allowed_values = np.arange(0, self.N + 10, 10)  # X-axis values (0, 10, 20, ..., 100)
        R0_values = self.calculate_R0(allowed_values)

        dfe_region = R0_values < 1
        ee_region = R0_values >= 1

        plt.figure(figsize=(10, 6))

        plt.plot(allowed_values, R0_values, label='$R_0$', color='black', linewidth=2)

        plt.fill_between(allowed_values, 0, R0_values, where=dfe_region, facecolor='blue', alpha=0.3,
                         label='DFE region ($R_0 < 1$)')
        plt.fill_between(allowed_values, 0, R0_values, where=ee_region, facecolor='red', alpha=0.3,
                         label='EE region ($R_0 \geq 1$)')

        plt.axhline(y=1, color='green', linestyle='--', label='$R_0 = 1$')

        plt.xlabel('Allowed Population $N_i$')
        plt.ylabel('$R_0$')
        plt.title(f'Threshold Behavior of $R_0$\nMethod: {method_label}\n(α_m = {self.alpha_m}, β = {self.beta})')
        plt.legend()
        plt.grid(True)
        plt.xlim(left=0, right=self.N)
        plt.ylim(bottom=0, top=np.max(R0_values) + 1)
        plt.tight_layout()
        plt.savefig(f'stoch_plots/threshold_behavior_R0_{method_label}.png')
        # plt.show()

    def get_reward(self, alpha: float):
        # If these attributes are not defined, initialize them based on some logic
        if not hasattr(self, 'allowed_students_per_course'):
            self.allowed_students_per_course = [random.randint(50, 100) for _ in range(10)]  # Example list

        if not hasattr(self, 'student_status'):
            self.student_status = [random.randint(0, 10) for _ in range(10)]  # Example list

        # Sum all allowed students and infected students across all courses
        total_allowed = sum(self.allowed_students_per_course)
        total_infected = sum(self.student_status)

        # Calculate the reward using the total values
        reward = int(alpha * total_allowed - (1 - alpha) * total_infected)

        return reward

    def animate_infection_dynamics(self, infected_values, community_risk_values, total_weeks, alpha, save_as_gif=False):
        fig, (ax_infection, ax_risk) = plt.subplots(2, 1, figsize=(8, 10))

        # Set up for the infection spread animation
        ax_infection.set_xlim(-10, 10)
        ax_infection.set_ylim(-10, 10)
        ax_infection.set_title("Infection Dynamics Over Time")

        # Draw a box to represent the controlled region
        box_boundary = plt.Rectangle((-5, -5), 10, 10, edgecolor='green', facecolor='none', lw=2)
        ax_infection.add_patch(box_boundary)

        # Inside region markers (blue) and outside region markers (red)
        inside_markers, = ax_infection.plot([], [], 'bo', markersize=5, label="Inside (Controlled)")
        outside_markers, = ax_infection.plot([], [], 'ro', markersize=5, label="Outside (Community)")

        # Set up for the community risk animation
        ax_risk.set_xlim(0, total_weeks)
        ax_risk.set_ylim(0, 1)
        ax_risk.set_title("Community Risk Over Time")
        ax_risk.set_xlabel("Weeks")
        ax_risk.set_ylabel("Risk Level")
        line, = ax_risk.plot([], [], lw=2, label='Community Risk')

        fig.tight_layout()

        # Initialize the infection markers' positions based on the number of infected individuals at the first time step
        positions = np.random.uniform(-5, 5, size=(100, 2))  # Start with 100 individuals

        # Infection boundary radius
        boundary_radius = 5.0

        def init():
            inside_markers.set_data([], [])
            outside_markers.set_data([], [])
            line.set_data([], [])
            return inside_markers, outside_markers, line

        def update(frame):
            # Infection model (inside interactions vs community interactions)
            infection_spread_inside = self.alpha_m * infected_values[frame] / self.max_infected
            infection_spread_community = self.beta * community_risk_values[frame] * infected_values[
                frame] ** 2 / self.max_infected

            # Update positions for inside infection spread
            positions[:, 0] += np.random.uniform(-0.5, 0.5, size=positions[:, 0].shape) * infection_spread_inside
            positions[:, 1] += np.random.uniform(-0.5, 0.5, size=positions[:, 1].shape) * infection_spread_inside

            # Update positions for community infection spread (outside)
            positions[:, 0] += np.random.uniform(-0.2, 0.2, size=positions[:, 0].shape) * infection_spread_community
            positions[:, 1] += np.random.uniform(-0.2, 0.2, size=positions[:, 1].shape) * infection_spread_community

            # Identify individuals inside and outside the boundary
            distances_from_center = np.linalg.norm(positions, axis=1)
            inside_indices = distances_from_center <= boundary_radius
            outside_indices = distances_from_center > boundary_radius

            # Set the data for inside and outside markers
            inside_markers.set_data(positions[inside_indices, 0], positions[inside_indices, 1])
            outside_markers.set_data(positions[outside_indices, 0], positions[outside_indices, 1])

            # Update community risk plot
            line.set_data(np.arange(frame + 1), community_risk_values[:frame + 1])

            # Display alpha value and calculate reward
            reward = self.get_reward(alpha)
            ax_infection.text(12, 8, f'Alpha: {alpha}', fontsize=12, color='purple')
            ax_infection.text(12, 6, f'Reward: {reward}', fontsize=12, color='green')

            # Update the text box showing the number of infected inside and outside
            ax_infection.text(12, 4, f'Infected Inside: {np.sum(inside_indices)}', fontsize=12, color='blue')
            ax_infection.text(12, 2, f'Infected Outside: {np.sum(outside_indices)}', fontsize=12, color='red')

            return inside_markers, outside_markers, line

        # Create the animation
        anim = FuncAnimation(fig, update, frames=total_weeks, init_func=init, blit=False, interval=500)

        # Save the animation as a GIF if specified
        if save_as_gif:
            anim.save('infection_dynamics.gif', writer='pillow', fps=2)

        # Show the animation
        plt.legend(loc='upper right')
        plt.show()

    def run_simulation(self, alpha):
        P_random, states_random = self.create_transition_matrix(lambda: self.calculate_R0(100))
        stationary_random = self.calculate_stationary_distribution(P_random)

        ergodic = self.check_ergodicity(P_random)
        print("The Markov chain is ergodic." if ergodic else "The Markov chain is not ergodic.")

        dfe_probability, ee_probability = self.analyze_stochastic_stability(stationary_random, states_random)
        # Simulate infection dynamics over time and community risk values for the animation
        # Load the community risk values from the CSV file
        data = pd.read_csv(csv_file_path)
        community_risk_values = data['Risk-Level'].values[:self.max_weeks]  # Use only required weeks

        infected_values = [np.random.randint(0, self.max_infected) for _ in range(self.max_weeks)]

        # Run the infection dynamics animation using the CSV data
        self.animate_infection_dynamics(infected_values, community_risk_values, self.max_weeks, alpha, save_as_gif=True)

        # self.plot_lyapunov_function_with_context(P_random, states_random, stationary_random, self.community_risk_method)
        # self.plot_stationary_distribution(stationary_random, states_random, self.community_risk_method)
        # # self.plot_transition_matrix_for_policy(P_random, self.community_risk_method, self.alpha, "myopic_policy")
        # self.plot_threshold_behavior_alpha_beta(self.community_risk_method)
        # self.plot_transition_matrix(P_random, self.community_risk_method)
        # self.plot_threshold_behavior_R0(self.community_risk_method)


        R0 = self.calculate_R0(100)
        max_R0 = np.max(R0)
        print(f"R0 (for maximum allowed population) = {max_R0:.2f}")

        stochastic_stability = "DFE" if dfe_probability > ee_probability else "EE"

        return {
            'Ergodic': ergodic,
            'DFE Probability': dfe_probability,
            'EE Probability': ee_probability,
            'R0 for max allowed': max_R0,
            'Stochastic Stability': stochastic_stability
        }
# Run the simulation for all methods
methods = ['uniform', 'data', 'sinusoidal']
results = []
# Assuming this script is inside a subfolder
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_file_path = os.path.join(root_directory, 'aggregated_weekly_risk_levels.csv')

for method in methods:
    print(f"Running simulation for method: {method}")
    alpha = 0.8
    sim = StochasticStabilitySimulation(
            N=100,
            alpha_m=0.005,
            beta=0.001,
            community_risk_method=method,
            community_risk_file=csv_file_path if method == 'data' else None,
            seed=42
        )
    result = sim.run_simulation(alpha)
    result['Method'] = method
    results.append(result)

# Create a DataFrame to summarize the results
results_df = pd.DataFrame(results)
print(results_df)

# Save the summary table to a CSV file
results_df.to_csv('stoch_plots/simulation_summary.csv', index=False)

