"""
Extra Credit 1: Exploration Strategies for Q-Learning on Blackjack

This script compares different exploration strategies:
1. Epsilon-greedy with various epsilon values (0.01, 0.05, 0.1, 0.2, 0.5)
2. Boltzmann (Softmax) exploration with various temperatures
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class QLearningAgentWithExploration:
    """
    Q-Learning Agent with multiple exploration strategies.
    """
    
    def __init__(self, learning_rate=0.1, discount_factor=1.0, 
                 exploration_strategy='epsilon_greedy', epsilon=0.1, temperature=1.0):
        """
        Initialize the Q-Learning agent.
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            exploration_strategy: 'epsilon_greedy' or 'boltzmann'
            epsilon: Exploration rate for epsilon-greedy
            temperature: Temperature for Boltzmann exploration
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.temperature = temperature
        
        # Q-table
        self.Q = defaultdict(lambda: np.zeros(2))
        
    def get_action(self, state, training=True):
        """
        Select action using the specified exploration strategy.
        """
        if not training:
            # Always greedy during evaluation
            return np.argmax(self.Q[state])
        
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy(state)
        elif self.exploration_strategy == 'boltzmann':
            return self._boltzmann(state)
        else:
            raise ValueError(f"Unknown strategy: {self.exploration_strategy}")
    
    def _epsilon_greedy(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.Q[state])
    
    def _boltzmann(self, state):
        """
        Boltzmann (Softmax) action selection.
        
        P(a) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
        """
        q_values = self.Q[state]
        
        # Handle numerical stability
        max_q = np.max(q_values)
        exp_values = np.exp((q_values - max_q) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        return np.random.choice(len(q_values), p=probabilities)
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning update."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state][action] += self.lr * (target - self.Q[state][action])


def train_agent(env, agent, num_episodes=50000, eval_interval=5000, eval_episodes=1000):
    """
    Train an agent and track performance.
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
            avg_reward = evaluate_agent(env, agent, num_episodes=eval_episodes)
            eval_rewards.append((episode + 1, avg_reward))
    
    return training_rewards, eval_rewards


def evaluate_agent(env, agent, num_episodes=1000):
    """Evaluate agent without exploration."""
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


def run_experiment():
    """Run all experiments and compare exploration strategies."""
    np.random.seed(42)
    
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    num_episodes = 100000
    
    print("=" * 70)
    print("Extra Credit 1: Exploration Strategy Comparison for Blackjack")
    print("=" * 70)
    
    # Configuration for experiments
    experiments = [
        # Epsilon-greedy experiments
        {'name': 'ε-greedy (ε=0.01)', 'strategy': 'epsilon_greedy', 'epsilon': 0.01},
        {'name': 'ε-greedy (ε=0.05)', 'strategy': 'epsilon_greedy', 'epsilon': 0.05},
        {'name': 'ε-greedy (ε=0.1)', 'strategy': 'epsilon_greedy', 'epsilon': 0.1},
        {'name': 'ε-greedy (ε=0.2)', 'strategy': 'epsilon_greedy', 'epsilon': 0.2},
        {'name': 'ε-greedy (ε=0.5)', 'strategy': 'epsilon_greedy', 'epsilon': 0.5},
        # Boltzmann experiments
        {'name': 'Boltzmann (τ=0.1)', 'strategy': 'boltzmann', 'temperature': 0.1},
        {'name': 'Boltzmann (τ=0.5)', 'strategy': 'boltzmann', 'temperature': 0.5},
        {'name': 'Boltzmann (τ=1.0)', 'strategy': 'boltzmann', 'temperature': 1.0},
        {'name': 'Boltzmann (τ=2.0)', 'strategy': 'boltzmann', 'temperature': 2.0},
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\nTraining: {exp['name']}...")
        
        if exp['strategy'] == 'epsilon_greedy':
            agent = QLearningAgentWithExploration(
                exploration_strategy='epsilon_greedy',
                epsilon=exp['epsilon']
            )
        else:
            agent = QLearningAgentWithExploration(
                exploration_strategy='boltzmann',
                temperature=exp['temperature']
            )
        
        training_rewards, eval_rewards = train_agent(
            env, agent, 
            num_episodes=num_episodes,
            eval_interval=10000,
            eval_episodes=5000
        )
        
        # Final evaluation
        final_avg = evaluate_agent(env, agent, num_episodes=10000)
        
        results[exp['name']] = {
            'training_rewards': training_rewards,
            'eval_rewards': eval_rewards,
            'final_avg': final_avg,
            'agent': agent
        }
        
        print(f"  Final Average Reward: {final_avg:.4f}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Final Avg Reward':>18}")
    print("-" * 45)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_avg'], reverse=True)
    for name, data in sorted_results:
        print(f"{name:<25} {data['final_avg']:>18.4f}")
    
    # Plot results
    plot_comparison(results, save_path="extra_credit_1_comparison.png")
    
    env.close()
    return results


def plot_comparison(results, save_path):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color maps for different strategies
    epsilon_colors = plt.cm.Blues(np.linspace(0.3, 0.9, 5))
    boltzmann_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, 4))
    
    # Plot 1: Learning curves (moving average)
    ax1 = axes[0]
    color_idx_eps = 0
    color_idx_bol = 0
    
    for name, data in results.items():
        rewards = data['training_rewards']
        # Calculate episode rewards from cumulative
        episode_rewards = np.diff(np.concatenate([[0], rewards]))
        
        # Moving average
        window = 2000
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            x = np.arange(window, len(episode_rewards) + 1)
            
            if 'ε-greedy' in name:
                color = epsilon_colors[color_idx_eps]
                color_idx_eps += 1
            else:
                color = boltzmann_colors[color_idx_bol]
                color_idx_bol += 1
            
            ax1.plot(x, moving_avg, label=name, alpha=0.8, color=color)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Moving Average Reward (window=2000)')
    ax1.set_title('Learning Curves by Exploration Strategy')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final performance bar chart
    ax2 = axes[1]
    names = list(results.keys())
    final_rewards = [results[n]['final_avg'] for n in names]
    
    colors = []
    for n in names:
        if 'ε-greedy' in n:
            colors.append('steelblue')
        else:
            colors.append('darkorange')
    
    bars = ax2.barh(names, final_rewards, color=colors, alpha=0.7)
    ax2.set_xlabel('Final Average Reward')
    ax2.set_title('Final Policy Performance Comparison')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, final_rewards):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {save_path}")


if __name__ == "__main__":
    results = run_experiment()
