import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import os

save_folder = './figures/agent_trajectories'
os.makedirs(save_folder, exist_ok=True)

ACTION_MAP = {
    0: 'Left',
    1: 'Right',
    2: 'Forward',
    3: 'Stay'
}

# Number of trajectories to generate
num_trajectories = 10  # Change this to generate more or fewer plots
# Run a random agent
num_steps = 2000

for trajectory_idx in range(num_trajectories):
    # Create the environment in full-render mode
    env = gym.make('MiniGrid-FourRooms-v0', render_mode='rgb_array')
    obs, info = env.reset()
    agent_positions = []

    
    for _ in range(num_steps):
        action = np.random.randint(3)
        obs, reward, terminated, truncated, info = env.step(action)
        agent_pos = env.unwrapped.agent_pos
        agent_positions.append(tuple(agent_pos))

    # Create a new figure for each trajectory
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 8))

    full_env_image = env.render()
    agent_positions = np.array(agent_positions)

    scale_x = full_env_image.shape[1] / env.unwrapped.width
    scale_y = full_env_image.shape[0] / env.unwrapped.height

    agent_coords = np.column_stack([
        agent_positions[:, 0] * scale_x + scale_x / 2,
        agent_positions[:, 1] * scale_y + scale_y / 2
    ])

    # Display the full environment image as the background
    ax.imshow(full_env_image)

    sc = ax.scatter(
        agent_coords[:, 0], agent_coords[:, 1],
        c=range(len(agent_positions)),
        cmap="cool", s=80, marker="o",
        edgecolor='gray', linewidth=0.5, zorder=3
    )

    ax.plot(
        agent_coords[:, 0], agent_coords[:, 1],
        color='white', linewidth=1, alpha=0.7, zorder=2
    )

    ax.scatter(
        agent_coords[0, 0], agent_coords[0, 1],
        c='red', s=150, marker="*", edgecolor='gold',
        linewidth=2, zorder=4, label="Start"
    )

    ax.scatter(
        agent_coords[-1, 0], agent_coords[-1, 1],
        c='lime', s=150, marker="X", edgecolor='darkgreen',
        linewidth=2, zorder=4, label="End"
    )

    ax.set_title(f"Agent Trajectory {trajectory_idx + 1} over {num_steps} Steps", 
                 fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Time Step', rotation=270, labelpad=15)
    ax.legend(fontsize=12, labelcolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'agent-trajectory-{num_steps}-{trajectory_idx+1}.png'), dpi=300)
    plt.close()  # Close the figure to free memory

env.close()  # Close the environment when done
