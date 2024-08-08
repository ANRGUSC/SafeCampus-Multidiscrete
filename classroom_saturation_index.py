import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Create a directory for the plots
os.makedirs('control_plots', exist_ok=True)

# Parameters
c_risk = 0.01  # Community risk
N = 100  # Total number of students
T = 50  # Time steps for simulation
alpha_m, beta = 0.05, 0.5  # Example parameters


def calculate_csi(alpha_m, beta, I_c, u):
    with np.errstate(divide='ignore', invalid='ignore'):
        csi = (alpha_m * I_c * u + beta * c_risk * u ** 2) / u
    return np.where(np.isfinite(csi), csi, 0)
def simulate_infections(alpha_m, beta, strategy='random'):
    I_c = np.zeros(T)
    I_c[0] = 1  # Start with one infected student
    u = np.zeros(T)

    for t in range(1, T):
        if strategy == 'random':
            u[t] = np.random.choice([0, 0.5 * N, N])
        elif strategy == 'continuous':
            u[t] = np.random.uniform(0, N)
        elif strategy == 'bang_bang':
            csi = calculate_csi(alpha_m, beta, I_c[t - 1], N)
            u[t] = N if csi < 1 else 1

        I_n = min(alpha_m * I_c[t - 1] * u[t] + beta * c_risk * u[t] ** 2, u[t])
        I_c[t] = min(I_c[t - 1] + I_n, u[t])  # Limit I_c to u[t]

    return I_c, u


# Visualize CSI for different alpha and beta
alpha_range = np.linspace(0, 0.1, 100)
beta_range = np.linspace(0, 1, 100)
alpha_mesh, beta_mesh = np.meshgrid(alpha_range, beta_range)

I_c = 10  # Assume 10 currently infected students
u = 50  # Assume 50 students allowed in class

csi_values = calculate_csi(alpha_mesh, beta_mesh, I_c, u)

plt.figure(figsize=(12, 10))
contour = plt.contourf(alpha_mesh, beta_mesh, csi_values, levels=20, cmap='viridis')
plt.colorbar(contour, label='CSI')

# Add contour line for CSI = 1
contour_line = plt.contour(alpha_mesh, beta_mesh, csi_values, levels=[1], colors='red', linestyles='dashed',
                           linewidths=2)
plt.clabel(contour_line, inline=True, fontsize=10, fmt='CSI = %.0f')

plt.xlabel('α (Transmission risk)')
plt.ylabel('β (Community risk scaling factor)')
plt.title('Classroom Saturation Index (CSI) for different α and β')

# Add legend for alpha and beta values
alpha_point = alpha_range[len(alpha_range) // 2]
beta_point = beta_range[len(beta_range) // 2]
plt.plot(alpha_point, beta_point, 'r*', markersize=10, label=f'α = {alpha_point:.3f}, β = {beta_point:.3f}')
plt.legend(loc='lower right')

plt.savefig('control_plots/csi_heatmap.png')
plt.close()

# Simulate infections with different strategies
strategies = ['random', 'continuous', 'bang_bang']
I_c_results = {}
u_results = {}

for strategy in strategies:
    I_c_results[strategy], u_results[strategy] = simulate_infections(alpha_m, beta, strategy)

# Plot infection dynamics as bar charts in horizontal subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))

for i, strategy in enumerate(strategies):
    axes[i].bar(range(T), I_c_results[strategy], color=colors[i])
    axes[i].set_title(f'{strategy.capitalize()} Strategy')
    axes[i].set_xlabel('Time Steps')
    axes[i].set_ylim(0, N)
    if i == 0:
        axes[i].set_ylabel('Number of Infected Students')

fig.suptitle(f'Infection Dynamics (α={alpha_m}, β={beta}, c_risk={c_risk})')
plt.savefig('control_plots/infection_dynamics_bars.png')
plt.close()
# # Plot control strategies
# plt.figure(figsize=(12, 6))
# for strategy in strategies:
#     plt.plot(range(T), u_results[strategy], label=f'{strategy.capitalize()} Strategy')
# plt.xlabel('Time Steps')
# plt.ylabel('Number of Students Allowed (u)')
# plt.title('Control Strategies Comparison')
# plt.legend()
# plt.savefig('control_plots/control_strategies.png')
# plt.close()

# Plot CSI threshold behavior
u_range = np.linspace(0, N, 100)
I_c_values = [10, 20, 30]  # Different levels of current infections

plt.figure(figsize=(12, 6))
for I_c in I_c_values:
    csi_values = calculate_csi(alpha_m, beta, I_c, u_range)
    plt.plot(u_range, csi_values, label=f'I_c = {I_c}')

plt.axhline(y=1, color='r', linestyle='--', label='CSI = 1')
plt.fill_between(u_range, 0, 1, alpha=0.2, color='blue', label='CSI < 1')
plt.fill_between(u_range, 1, max(np.max(csi_values), 2), alpha=0.2, color='red', label='CSI > 1')

plt.xlabel('Number of Students Allowed (u)')
plt.ylabel('Classroom Saturation Index (CSI)')
plt.title('CSI Threshold Behavior')
plt.legend()
plt.ylim(0, max(np.max(csi_values), 2))
plt.savefig('control_plots/csi_threshold.png')
plt.close()

