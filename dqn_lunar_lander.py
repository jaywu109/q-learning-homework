"""
DQN for Lunar Lander
Homework 4 - Part 2

This script implements a DQN agent to learn a policy for Lunar Lander.
Based on the PyTorch DQN tutorial with modifications for Lunar Lander.
"""

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Set up matplotlib
matplotlib.use('Agg')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Transition namedtuple for storing experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    Deep Q-Network with fully connected layers.
    
    For Lunar Lander:
    - Input: 8 state features (x, y, vel_x, vel_y, angle, angular_vel, left_leg, right_leg)
    - Output: 4 Q-values (no-op, fire left, fire main, fire right)
    """
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """
    DQN Agent with experience replay and target network.
    
    Uses soft target updates (Polyak averaging) instead of hard updates.
    Uses Huber loss instead of MSE for more stable training.
    """
    
    def __init__(self, n_observations, n_actions, 
                 batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05, 
                 eps_decay=1000, tau=0.005, lr=1e-4, memory_size=10000):
        """
        Initialize the DQN agent.
        
        Args:
            n_observations: Size of state space
            n_actions: Number of actions
            batch_size: Size of training batches
            gamma: Discount factor
            eps_start: Starting epsilon for exploration
            eps_end: Final epsilon for exploration  
            eps_decay: Decay rate for epsilon
            tau: Soft update coefficient for target network
            lr: Learning rate
            memory_size: Size of replay buffer
        """
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        
        # Networks
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Step counter for epsilon decay
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            training: Whether in training mode (use exploration)
            
        Returns:
            action: Selected action as tensor
        """
        if training:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], 
                                   device=device, dtype=torch.long)
        else:
            # Greedy action selection during evaluation
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
    
    def optimize(self):
        """
        Perform one step of optimization on the policy network.
        """
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Compute mask for non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                       device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update(self):
        """
        Soft update of the target network's weights:
        θ′ ← τθ + (1−τ)θ′
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + \
                                          target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


def train_dqn(env, agent, num_episodes=500, max_steps_per_episode=1000):
    """
    Train the DQN agent.
    
    Args:
        env: Gymnasium environment
        agent: DQNAgent instance
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        episode_durations: List of episode lengths
        episode_rewards: List of episode total rewards
    """
    episode_durations = []
    episode_rewards = []
    
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        
        for t in count():
            action = agent.select_action(state, training=True)
            observation, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            done = terminated or truncated
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Store transition
            agent.memory.push(state, action, next_state, reward_tensor)
            
            state = next_state
            
            # Optimize
            agent.optimize()
            
            # Soft update target network
            agent.soft_update()
            
            if done or t >= max_steps_per_episode:
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                break
        
        # Print progress
        if (i_episode + 1) % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            avg_duration = np.mean(episode_durations[-25:])
            print(f"Episode {i_episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 25): {avg_reward:.2f}, "
                  f"Avg Duration: {avg_duration:.1f}, "
                  f"Epsilon: {agent.eps_end + (agent.eps_start - agent.eps_end) * math.exp(-1. * agent.steps_done / agent.eps_decay):.3f}")
    
    return episode_durations, episode_rewards


def evaluate_policy(env, agent, num_episodes=100, render=False):
    """
    Evaluate the learned policy without exploration.
    
    Args:
        env: Gymnasium environment
        agent: DQNAgent instance
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        
    Returns:
        avg_reward: Average reward per episode
        rewards: List of episode rewards
    """
    rewards = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            observation, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            
            if not done:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        rewards.append(total_reward)
    
    return np.mean(rewards), rewards


def random_policy_baseline(env, num_episodes=100):
    """
    Evaluate a random policy as a baseline.
    
    Args:
        env: Gymnasium environment
        num_episodes: Number of episodes
        
    Returns:
        avg_reward: Average reward per episode
        rewards: List of episode rewards
    """
    rewards = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        rewards.append(total_reward)
    
    return np.mean(rewards), rewards


def plot_training_results(episode_rewards, random_avg, save_path="lunar_lander_training.png"):
    """
    Plot training results.
    
    Args:
        episode_rewards: List of episode rewards during training
        random_avg: Average reward for random policy
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Episode rewards
    ax1 = axes[0]
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.4, label='Episode Reward')
    
    # Moving average
    window = 25
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window, len(episode_rewards) + 1), moving_avg, 
                 label=f'Moving Avg (window={window})', linewidth=2)
    
    ax1.axhline(y=random_avg, color='r', linestyle='--', label=f'Random Policy: {random_avg:.1f}')
    ax1.axhline(y=200, color='g', linestyle=':', label='Solved Threshold: 200')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('DQN Training Progress on Lunar Lander')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rewards
    ax2 = axes[1]
    cumulative = np.cumsum(episode_rewards)
    ax2.plot(episodes, cumulative, label='DQN Cumulative Reward')
    
    # Compare with random (extrapolated)
    random_cumulative = np.arange(1, len(episode_rewards) + 1) * random_avg
    ax2.plot(episodes, random_cumulative, 'r--', label='Random Policy (linear projection)')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Reward Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {save_path}")


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 60)
    print("Part 2: DQN for Lunar Lander")
    print("=" * 60)
    
    # Create Lunar Lander environment (no rendering during training)
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    
    # Get environment dimensions
    n_observations = env.observation_space.shape[0]  # 8 state features
    n_actions = env.action_space.n  # 4 actions
    
    print(f"\nEnvironment: LunarLander-v3")
    print(f"State space: {n_observations} dimensions")
    print(f"Action space: {n_actions} discrete actions")
    
    # Initialize DQN agent with tuned hyperparameters
    agent = DQNAgent(
        n_observations=n_observations,
        n_actions=n_actions,
        batch_size=128,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=1000,
        tau=0.005,
        lr=1e-4,
        memory_size=10000
    )
    
    # Evaluate random policy baseline first
    print("\nEvaluating random policy baseline...")
    random_avg, random_rewards = random_policy_baseline(env, num_episodes=100)
    print(f"Random Policy Average Reward: {random_avg:.2f}")
    
    # Train DQN agent
    print("\nTraining DQN agent...")
    num_episodes = 500
    episode_durations, episode_rewards = train_dqn(
        env, agent, 
        num_episodes=num_episodes,
        max_steps_per_episode=1000
    )
    
    # Final evaluation
    print("\nFinal evaluation of learned DQN policy...")
    final_avg, final_rewards = evaluate_policy(env, agent, num_episodes=100)
    print(f"Learned Policy Average Reward: {final_avg:.2f}")
    
    # Results summary
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"Random Policy Average Reward: {random_avg:.2f}")
    print(f"DQN Policy Average Reward: {final_avg:.2f}")
    print(f"Improvement: {final_avg - random_avg:.2f} points")
    
    # Landing success rate (reward > 100 is typically a successful landing)
    successful_landings = sum(1 for r in final_rewards if r > 100)
    print(f"\nSuccessful Landings (reward > 100): {successful_landings}%")
    
    # Best performance
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"Last 25 Episodes Average: {np.mean(episode_rewards[-25:]):.2f}")
    
    # Plot training results
    print("\nGenerating plots...")
    plot_training_results(episode_rewards, random_avg, 
                          save_path="lunar_lander_training.png")
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), 'lunar_lander_dqn.pth')
    print("Model saved to lunar_lander_dqn.pth")
    
    env.close()
    
    return agent, episode_rewards, random_avg, final_avg


if __name__ == "__main__":
    agent, episode_rewards, random_avg, final_avg = main()
