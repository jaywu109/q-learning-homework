"""
Tabular Q-Learning for Blackjack
Homework 4 - Part 1

This script implements a vanilla Q-Learning agent to learn to play Blackjack.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning Agent for Blackjack.
    
    The state space is a tuple of (player_sum, dealer_card, usable_ace).
    - player_sum: 4-21 (values below 12 are not relevant since hitting is always safe)
    - dealer_card: 1-10 (Ace=1, face cards=10)
    - usable_ace: True/False
    
    Action space:
    - 0: Stand (stick)
    - 1: Hit
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=1.0, epsilon=0.1):
        """
        Initialize the Q-Learning agent.
        
        Args:
            learning_rate (float): Learning rate (alpha) for Q-value updates
            discount_factor (float): Discount factor (gamma) for future rewards
            epsilon (float): Exploration rate for epsilon-greedy policy
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table as a dictionary of state -> action values
        # defaultdict automatically initializes new states with zeros
        self.Q = defaultdict(lambda: np.zeros(2))
        
    def get_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
            training: If True, use epsilon-greedy; if False, use greedy policy
            
        Returns:
            action: 0 (stand) or 1 (hit)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(0, 2)
        else:
            # Exploit: choose best action based on Q-values
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is finished
        """
        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: include future Q-value estimate
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # Q-learning update
        self.Q[state][action] += self.lr * (target - self.Q[state][action])
    
    def get_policy(self):
        """
        Extract the greedy policy from Q-values.
        
        Returns:
            policy: Dictionary mapping states to best actions
        """
        policy = {}
        for state, q_values in self.Q.items():
            policy[state] = np.argmax(q_values)
        return policy


def train_q_learning(env, agent, num_episodes=100000, eval_interval=1000, eval_episodes=100):
    """
    Train the Q-learning agent.
    
    Args:
        env: Gymnasium environment
        agent: QLearningAgent instance
        num_episodes: Number of training episodes
        eval_interval: Evaluate policy every N episodes
        eval_episodes: Number of episodes for each evaluation
        
    Returns:
        training_rewards: List of cumulative rewards during training
        eval_rewards: List of (episode, avg_reward) tuples for evaluation
    """
    training_rewards = []
    eval_rewards = []
    cumulative_reward = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        cumulative_reward += episode_reward
        training_rewards.append(cumulative_reward)
        
        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            avg_reward = evaluate_policy(env, agent, num_episodes=eval_episodes)
            eval_rewards.append((episode + 1, avg_reward))
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Cumulative Reward: {cumulative_reward:.1f}, "
                  f"Avg Eval Reward: {avg_reward:.3f}")
    
    return training_rewards, eval_rewards


def evaluate_policy(env, agent, num_episodes=100):
    """
    Evaluate the current learned policy (no exploration).
    
    Args:
        env: Gymnasium environment
        agent: QLearningAgent instance
        num_episodes: Number of episodes to evaluate
        
    Returns:
        avg_reward: Average reward per episode
    """
    total_reward = 0
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
    
    return total_reward / num_episodes


def random_policy_baseline(env, num_episodes=10000):
    """
    Evaluate a random policy as a baseline.
    
    Args:
        env: Gymnasium environment
        num_episodes: Number of episodes to run
        
    Returns:
        avg_reward: Average reward per episode
        cumulative_rewards: List of cumulative rewards over episodes
    """
    cumulative_rewards = []
    cumulative_reward = 0
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        cumulative_reward += episode_reward
        cumulative_rewards.append(cumulative_reward)
    
    avg_reward = cumulative_reward / num_episodes
    return avg_reward, cumulative_rewards


def plot_learning_curves(q_rewards, random_rewards, save_path="blackjack_learning_curve.png"):
    """
    Plot learning curves comparing Q-learning with random policy.
    
    Args:
        q_rewards: List of cumulative rewards for Q-learning
        random_rewards: List of cumulative rewards for random policy
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cumulative rewards over episodes
    ax1 = axes[0]
    episodes = np.arange(1, len(q_rewards) + 1)
    ax1.plot(episodes, q_rewards, label='Q-Learning', alpha=0.8)
    
    # Extend random rewards to match Q-learning length if needed
    if len(random_rewards) < len(q_rewards):
        # Extrapolate random policy
        random_extension = np.linspace(random_rewards[-1], 
                                        random_rewards[-1] + (len(q_rewards) - len(random_rewards)) * (random_rewards[-1] / len(random_rewards)),
                                        len(q_rewards) - len(random_rewards))
        random_extended = np.concatenate([random_rewards, random_extension])
    else:
        random_extended = random_rewards[:len(q_rewards)]
    
    ax1.plot(episodes, random_extended, label='Random Policy', alpha=0.8, linestyle='--')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average reward (smoothed performance)
    ax2 = axes[1]
    window = 1000
    q_rewards_array = np.array(q_rewards)
    random_rewards_array = np.array(random_extended)
    
    # Calculate episode rewards from cumulative
    q_episode_rewards = np.diff(np.concatenate([[0], q_rewards_array]))
    random_episode_rewards = np.diff(np.concatenate([[0], random_rewards_array]))
    
    # Moving average
    q_moving_avg = np.convolve(q_episode_rewards, np.ones(window)/window, mode='valid')
    random_moving_avg = np.convolve(random_episode_rewards, np.ones(window)/window, mode='valid')
    
    ax2.plot(np.arange(window, len(q_rewards) + 1), q_moving_avg, label='Q-Learning', alpha=0.8)
    ax2.plot(np.arange(window, len(q_rewards) + 1), random_moving_avg, label='Random Policy', alpha=0.8, linestyle='--')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel(f'Average Reward (window={window})')
    ax2.set_title('Moving Average Reward Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved to {save_path}")


def plot_evaluation_curve(eval_rewards, random_avg, save_path="blackjack_evaluation.png"):
    """
    Plot evaluation performance over training.
    
    Args:
        eval_rewards: List of (episode, avg_reward) tuples
        random_avg: Average reward for random policy
        save_path: Path to save the figure
    """
    episodes, rewards = zip(*eval_rewards)
    
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, 'b-o', label='Q-Learning Policy', markersize=4)
    plt.axhline(y=random_avg, color='r', linestyle='--', label=f'Random Policy (avg={random_avg:.3f})')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward (over 100 evaluation episodes)')
    plt.title('Policy Evaluation During Q-Learning Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation curve saved to {save_path}")


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create Blackjack environment
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    
    print("=" * 60)
    print("Part 1: Tabular Q-Learning for Blackjack")
    print("=" * 60)
    
    # Initialize Q-learning agent with tuned hyperparameters
    # For Blackjack, discount factor of 1.0 is appropriate since episodes are short
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=1.0,
        epsilon=0.1
    )
    
    # Train the agent
    print("\nTraining Q-Learning agent...")
    num_episodes = 100000
    training_rewards, eval_rewards = train_q_learning(
        env, agent, 
        num_episodes=num_episodes,
        eval_interval=5000,
        eval_episodes=1000
    )
    
    # Evaluate random policy baseline
    print("\nEvaluating random policy baseline...")
    random_avg, random_cumulative = random_policy_baseline(env, num_episodes=num_episodes)
    print(f"Random Policy Average Reward: {random_avg:.4f}")
    
    # Final evaluation of learned policy
    print("\nFinal evaluation of learned Q-Learning policy...")
    final_avg = evaluate_policy(env, agent, num_episodes=10000)
    print(f"Learned Policy Average Reward: {final_avg:.4f}")
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Results Summary:")
    print("=" * 60)
    print(f"Random Policy Average Reward: {random_avg:.4f}")
    print(f"Q-Learning Policy Average Reward: {final_avg:.4f}")
    print(f"Improvement: {((final_avg - random_avg) / abs(random_avg) * 100):.1f}%")
    
    # Win/loss/draw rates
    print("\nCalculating win/loss/draw rates...")
    wins, losses, draws = 0, 0, 0
    for _ in range(10000):
        state, info = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    
    print(f"Win Rate: {wins/100:.1f}%")
    print(f"Loss Rate: {losses/100:.1f}%")
    print(f"Draw Rate: {draws/100:.1f}%")
    
    # Plot learning curves
    print("\nGenerating plots...")
    plot_learning_curves(training_rewards, random_cumulative, 
                         save_path="blackjack_learning_curve.png")
    plot_evaluation_curve(eval_rewards, random_avg,
                          save_path="blackjack_evaluation.png")
    
    # Save trained Q-table
    with open('blackjack_qtable.pkl', 'wb') as f:
        pickle.dump(dict(agent.Q), f)
    print("Q-table saved to blackjack_qtable.pkl")
    
    env.close()
    
    return agent, training_rewards, eval_rewards, random_avg, final_avg


if __name__ == "__main__":
    agent, training_rewards, eval_rewards, random_avg, final_avg = main()
