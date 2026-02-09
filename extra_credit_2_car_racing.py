"""
Extra Credit 2: CNN-based DQN for Car Racing

This script implements a DQN agent with convolutional layers to learn
from raw pixel input for the Car Racing environment.

Based on:
- https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN
- https://github.com/CCS-Lab/project_car_racing
"""

import gymnasium as gym
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

matplotlib.use('Agg')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Transition namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    """Experience replay buffer."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def preprocess_frame(frame):
    """
    Preprocess the frame for the CNN.
    - Convert to grayscale
    - Crop to focus on the road
    - Normalize to [0, 1]
    - Resize if needed
    
    Input: (96, 96, 3) RGB image
    Output: (84, 84) grayscale image
    """
    # Convert to grayscale
    gray = np.mean(frame, axis=2)
    
    # Crop to remove HUD at bottom (focus on main gameplay area)
    # Original is 96x96, crop to 84x84 from top-left
    cropped = gray[:84, 6:90]  # 84x84
    
    # Normalize
    normalized = cropped / 255.0
    
    return normalized.astype(np.float32)


class FrameStack:
    """Stack multiple frames for temporal information."""
    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque([], maxlen=num_frames)
    
    def reset(self, frame):
        """Reset with initial frame repeated."""
        for _ in range(self.num_frames):
            self.frames.append(frame)
        return self._get_state()
    
    def step(self, frame):
        """Add new frame and return stacked state."""
        self.frames.append(frame)
        return self._get_state()
    
    def _get_state(self):
        """Return stacked frames as (num_frames, H, W) array."""
        return np.stack(self.frames, axis=0)


class ConvDQN(nn.Module):
    """
    Convolutional DQN for pixel input.
    
    Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
    Output: Q-values for 5 discrete actions
    """
    def __init__(self, n_actions=5):
        super(ConvDQN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions
        # Input: 84x84
        # After conv1 (8, 4): (84-8)/4 + 1 = 20
        # After conv2 (4, 2): (20-4)/2 + 1 = 9
        # After conv3 (3, 1): (9-3)/1 + 1 = 7
        # Final: 64 * 7 * 7 = 3136
        
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CarRacingDQNAgent:
    """DQN Agent for Car Racing with CNN."""
    
    # Define discrete actions
    # 0: No action
    # 1: Steer left
    # 2: Steer right
    # 3: Gas
    # 4: Brake
    ACTIONS = [
        np.array([0.0, 0.0, 0.0]),   # No action
        np.array([-1.0, 0.0, 0.0]),  # Steer left
        np.array([1.0, 0.0, 0.0]),   # Steer right
        np.array([0.0, 1.0, 0.0]),   # Gas
        np.array([0.0, 0.0, 0.8]),   # Brake
    ]
    
    def __init__(self, batch_size=32, gamma=0.99, eps_start=1.0, eps_end=0.05,
                 eps_decay=10000, tau=0.005, lr=1e-4, memory_size=10000):
        self.n_actions = len(self.ACTIONS)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        
        # Networks
        self.policy_net = ConvDQN(self.n_actions).to(device)
        self.target_net = ConvDQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0
        self.frame_stack = FrameStack(num_frames=4)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy."""
        if training:
            sample = np.random.random()
            eps = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            
            if sample > eps:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    return self.policy_net(state_tensor).max(1).indices.item()
            else:
                return np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                return self.policy_net(state_tensor).max(1).indices.item()
    
    def get_continuous_action(self, action_idx):
        """Convert discrete action index to continuous action."""
        return self.ACTIONS[action_idx]
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, next_state, reward, done)
    
    def optimize(self):
        """Perform one optimization step."""
        if len(self.memory) < self.batch_size:
            return None
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
        done_batch = torch.tensor(batch.done, dtype=torch.bool, device=device)
        
        # For next states, we need to handle terminal states
        non_final_mask = ~done_batch
        non_final_next_states = torch.tensor(
            np.array([s for s, d in zip(batch.next_state, batch.done) if not d]),
            dtype=torch.float32, device=device
        )
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_next_states.numel() > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
    
    def soft_update(self):
        """Soft update target network."""
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.tau + \
                                      target_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_state_dict)


def train_car_racing(agent, num_episodes=200, max_steps=1000, skip_frames=4):
    """
    Train the DQN agent on Car Racing.
    
    Args:
        agent: CarRacingDQNAgent instance
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        skip_frames: Number of frames to skip (repeat action)
    """
    env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Preprocess and initialize frame stack
        frame = preprocess_frame(obs)
        state = agent.frame_stack.reset(frame)
        
        total_reward = 0
        negative_reward_counter = 0
        
        for step in range(max_steps):
            # Select action
            action_idx = agent.select_action(state, training=True)
            action = agent.get_continuous_action(action_idx)
            
            # Execute action for skip_frames
            reward_sum = 0
            done = False
            for _ in range(skip_frames):
                next_obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                done = terminated or truncated
                if done:
                    break
            
            # Preprocess next frame
            next_frame = preprocess_frame(next_obs)
            next_state = agent.frame_stack.step(next_frame)
            
            # Store transition
            agent.store_transition(state, action_idx, next_state, reward_sum, done)
            
            # Optimize
            agent.optimize()
            agent.soft_update()
            
            state = next_state
            total_reward += reward_sum
            
            # Early stopping if stuck
            if reward_sum < 0:
                negative_reward_counter += 1
                if negative_reward_counter > 25:
                    break
            else:
                negative_reward_counter = 0
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            eps = agent.eps_end + (agent.eps_start - agent.eps_end) * \
                np.exp(-1. * agent.steps_done / agent.eps_decay)
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {total_reward:.1f}, "
                  f"Avg (last 10): {avg_reward:.1f}, "
                  f"Epsilon: {eps:.3f}")
    
    env.close()
    return episode_rewards


def evaluate_random_policy(num_episodes=20, max_steps=500):
    """Evaluate random policy baseline."""
    env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
    
    actions = [
        np.array([0.0, 0.0, 0.0]),   # No action
        np.array([-1.0, 0.0, 0.0]),  # Steer left
        np.array([1.0, 0.0, 0.0]),   # Steer right
        np.array([0.0, 1.0, 0.0]),   # Gas
        np.array([0.0, 0.0, 0.8]),   # Brake
    ]
    
    rewards = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        
        for _ in range(max_steps):
            action = random.choice(actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
    
    env.close()
    return np.mean(rewards), rewards


def evaluate_policy(agent, num_episodes=20, max_steps=1000):
    """Evaluate learned policy."""
    env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
    
    rewards = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        
        frame = preprocess_frame(obs)
        state = agent.frame_stack.reset(frame)
        
        total_reward = 0
        
        for _ in range(max_steps):
            action_idx = agent.select_action(state, training=False)
            action = agent.get_continuous_action(action_idx)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            frame = preprocess_frame(next_obs)
            state = agent.frame_stack.step(frame)
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
    
    env.close()
    return np.mean(rewards), rewards


def plot_training(episode_rewards, random_avg, save_path="extra_credit_2_car_racing.png"):
    """Plot training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Episode rewards
    ax1 = axes[0]
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.4, label='Episode Reward')
    
    # Moving average
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window, len(episode_rewards) + 1), moving_avg,
                 label=f'Moving Avg (window={window})', linewidth=2)
    
    ax1.axhline(y=random_avg, color='r', linestyle='--', label=f'Random: {random_avg:.1f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('CNN-DQN Training on Car Racing')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative rewards
    ax2 = axes[1]
    cumulative = np.cumsum(episode_rewards)
    ax2.plot(episodes, cumulative, label='DQN Cumulative')
    random_cumulative = episodes * random_avg
    ax2.plot(episodes, random_cumulative, 'r--', label='Random (projected)')
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
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("Extra Credit 2: CNN-based DQN for Car Racing")
    print("=" * 70)
    
    # Random baseline
    print("\nEvaluating random policy baseline...")
    random_avg, random_rewards = evaluate_random_policy(num_episodes=20)
    print(f"Random Policy Average Reward: {random_avg:.2f}")
    
    # Create agent
    agent = CarRacingDQNAgent(
        batch_size=32,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=5000,
        tau=0.005,
        lr=1e-4,
        memory_size=10000
    )
    
    # Train
    print("\nTraining CNN-DQN agent...")
    num_episodes = 150  # Reduced for reasonable training time
    episode_rewards = train_car_racing(agent, num_episodes=num_episodes)
    
    # Final evaluation
    print("\nFinal evaluation of learned policy...")
    final_avg, final_rewards = evaluate_policy(agent, num_episodes=20)
    print(f"Learned Policy Average Reward: {final_avg:.2f}")
    
    # Results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Random Policy Average Reward: {random_avg:.2f}")
    print(f"DQN Policy Average Reward: {final_avg:.2f}")
    print(f"Improvement: {final_avg - random_avg:.2f} points")
    print(f"Best Episode Reward: {max(episode_rewards):.2f}")
    print(f"Last 10 Episodes Average: {np.mean(episode_rewards[-10:]):.2f}")
    
    # Plot
    plot_training(episode_rewards, random_avg)
    
    # Save model
    torch.save(agent.policy_net.state_dict(), 'car_racing_cnn_dqn.pth')
    print("Model saved to car_racing_cnn_dqn.pth")
    
    return agent, episode_rewards, random_avg, final_avg


if __name__ == "__main__":
    agent, episode_rewards, random_avg, final_avg = main()
