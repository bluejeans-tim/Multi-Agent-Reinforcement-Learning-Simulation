import os
import sys
import wandb
from wandb.integration.sb3 import WandbCallback
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from pettingzoo.utils.conversions import aec_to_parallel
import matplotlib.pyplot as plt
import numpy as np

# Dynamically add the project root to the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.environment import env  # Import after updating sys.path

# Define Early Stopping Callback
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, reward_threshold, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self):
        if self.locals["rewards"].mean() >= self.reward_threshold:
            print("Stopping training early as reward threshold reached.")
            return False  # Stop training
        return True

# Reward Tracking Callback
class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Check if the episode is done and log rewards
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.rewards.append(info["episode"]["r"])
        return True

# Define paths for logs and outputs
log_path = os.path.join(ROOT_DIR, 'logs', 'tensorboard')
model_save_dir = os.path.join(ROOT_DIR, 'models')
output_dir = os.path.join(ROOT_DIR, 'outputs')

# Ensure directories exist
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Initialize W&B
wandb.init(
    project="multi-agent-rl-simulation",
    config={
        "num_agents": 10,
        "max_cycles": 1000,
        "algorithm": "PPO",
        "batch_size": 256,
        "learning_rate": 3e-4,
        "total_timesteps": 50000,
        "iterations": 10
    },
    sync_tensorboard=True  # Sync TensorBoard logs with W&B
)

# Initialize the environment
def init_environment(num_agents, max_cycles):
    environment = env(num_agents=num_agents, max_cycles=max_cycles)  # AEC environment
    environment = aec_to_parallel(environment)  # Convert to ParallelEnv
    environment = ss.multiagent_wrappers.pad_observations_v0(environment)
    environment = ss.pettingzoo_env_to_vec_env_v1(environment)
    environment = ss.concat_vec_envs_v1(environment, 1, base_class="stable_baselines3")
    return VecMonitor(environment, log_path)

def train_model(model_name, num_agents, max_cycles, batch_size, learning_rate, total_timesteps, iterations):
    environment = init_environment(num_agents, max_cycles)
    model = PPO(
        "MlpPolicy",
        environment,
        verbose=1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tensorboard_log=log_path
    )

    # Define callbacks
    wandb_callback = WandbCallback(
        gradient_save_freq=100,  # Log gradients every 100 steps
        model_save_path=model_save_dir,  # Save models to W&B
        verbose=1
    )
    early_stopping = EarlyStoppingCallback(reward_threshold=100)

    # Track metrics for plotting
    rewards = []
    iterations_list = []

    for iters in range(iterations):
        print(f"Starting iteration {iters}...")
        reward_callback = RewardTrackingCallback()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[wandb_callback, early_stopping, reward_callback],
            reset_num_timesteps=False
        )

        # Calculate mean reward after each iteration
        if reward_callback.rewards:
            mean_reward = np.mean(reward_callback.rewards)
            rewards.append(mean_reward)
            wandb.log({"mean_reward": mean_reward})

        iterations_list.append(iters)

        # Save the model
        model_save_path = os.path.join(model_save_dir, f"{model_name}_{iters}.zip")
        model.save(model_save_path)
        print(f"Iteration {iters} complete. Model saved at {model_save_path}. Mean reward: {mean_reward}")

    # Plot and log rewards
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, rewards, label="Mean Reward", color="blue")
    plt.fill_between(
        iterations_list,
        np.array(rewards) - np.std(rewards),
        np.array(rewards) + np.std(rewards),
        color="blue",
        alpha=0.3,  # Transparency
        label="Confidence Interval"
    )
    plt.title("Training Rewards Over Iterations", fontsize=16)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plot_path = os.path.join(output_dir, f"reward_plot_{model_name}.png")
    plt.savefig(plot_path)
    plt.show()

    wandb.log({"Reward Plot": wandb.Image(plot_path)})  # Log plot to W&B

    # Close the environment
    environment.close()

# Train models for different formations
train_model("circle", num_agents=wandb.config.num_agents, max_cycles=wandb.config.max_cycles, batch_size=wandb.config.batch_size, learning_rate=wandb.config.learning_rate, total_timesteps=wandb.config.total_timesteps, iterations=wandb.config.iterations)
train_model("rectangle", num_agents=wandb.config.num_agents, max_cycles=wandb.config.max_cycles, batch_size=wandb.config.batch_size, learning_rate=wandb.config.learning_rate, total_timesteps=wandb.config.total_timesteps, iterations=wandb.config.iterations)

# Finish W&B run
wandb.finish()
