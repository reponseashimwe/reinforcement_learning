"""
REINFORCE Algorithm Implementation (Vanilla Policy Gradient)

This is a from-scratch PyTorch implementation of the REINFORCE algorithm
(Williams, 1992) with optional value function baseline for variance reduction.

Author: [Your Name]
Date: 2025-11-21
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import os
from pathlib import Path


class PolicyNetwork(nn.Module):
    """
    Policy network (actor) for discrete action spaces.

    Outputs a probability distribution over actions given an observation.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        Initialize policy network.

        Args:
            obs_dim: Observation space dimensionality
            action_dim: Action space dimensionality
            hidden_dims: List of hidden layer sizes
        """
        super(PolicyNetwork, self).__init__()

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.

        Args:
            obs: Observation tensor [batch_size, obs_dim]

        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.network(obs)

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: Observation array
            deterministic: If True, return argmax action

        Returns:
            Tuple of (action, log_prob)
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        logits = self.forward(obs_tensor)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action])
        else:
            dist = Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))

        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Value network (critic/baseline) for state value estimation.

    Used to reduce variance in policy gradient estimates.
    """

    def __init__(self, obs_dim: int, hidden_dims: List[int]):
        """
        Initialize value network.

        Args:
            obs_dim: Observation space dimensionality
            hidden_dims: List of hidden layer sizes
        """
        super(ValueNetwork, self).__init__()

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.

        Args:
            obs: Observation tensor [batch_size, obs_dim]

        Returns:
            State values [batch_size, 1]
        """
        return self.network(obs).squeeze(-1)


class REINFORCE:
    """
    REINFORCE algorithm (vanilla policy gradient with optional baseline).

    The algorithm:
    1. Collect full episode trajectory
    2. Compute discounted returns (Monte Carlo)
    3. Update policy to maximize expected return
    4. Optionally use value function baseline to reduce variance
    """

    def __init__(
        self,
        env,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        hidden_dims: List[int] = [128, 128],
        use_baseline: bool = False,
        max_grad_norm: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize REINFORCE algorithm.

        Args:
            env: Gymnasium environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            hidden_dims: Hidden layer dimensions
            use_baseline: Whether to use value function baseline
            max_grad_norm: Maximum gradient norm for clipping
            entropy_coef: Entropy bonus coefficient
            device: Device to run on ("cpu" or "cuda")
        """
        self.env = env
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.use_baseline = use_baseline
        self.device = device

        # Get environment dimensions
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Initialize policy network
        self.policy = PolicyNetwork(
            self.obs_dim,
            self.action_dim,
            hidden_dims
        ).to(device)

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

        # Initialize value network if using baseline
        if use_baseline:
            self.value = ValueNetwork(
                self.obs_dim,
                hidden_dims
            ).to(device)

            self.value_optimizer = optim.Adam(
                self.value.parameters(),
                lr=learning_rate
            )
        else:
            self.value = None
            self.value_optimizer = None

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []

    def collect_episode(self, deterministic: bool = False) -> Tuple[List, List, List, List]:
        """
        Collect a full episode trajectory.

        Args:
            deterministic: If True, use deterministic policy

        Returns:
            Tuple of (observations, actions, rewards, log_probs)
        """
        observations = []
        actions = []
        rewards = []
        log_probs = []

        obs, _ = self.env.reset()
        done = False

        while not done:
            # Store observation
            observations.append(obs)

            # Get action from policy
            action, log_prob = self.policy.get_action(obs, deterministic=deterministic)

            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            obs = next_obs

        return observations, actions, rewards, log_probs

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted returns (rewards-to-go).

        Args:
            rewards: List of rewards from episode

        Returns:
            Tensor of returns for each timestep
        """
        returns = []
        G = 0

        # Compute returns in reverse (from end to start)
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns (helps stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self, observations: List, actions: List, rewards: List, log_probs: List):
        """
        Update policy (and value network if using baseline).

        Args:
            observations: List of observations
            actions: List of actions taken
            rewards: List of rewards received
            log_probs: List of log probabilities
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        log_probs_tensor = torch.stack(log_probs).to(self.device)

        # Compute returns
        returns = self.compute_returns(rewards)

        # Compute advantages
        if self.use_baseline:
            # Use value function as baseline
            values = self.value(obs_tensor).detach()
            advantages = returns - values
        else:
            # No baseline, use raw returns
            advantages = returns

        # Compute policy loss
        policy_loss = -(log_probs_tensor * advantages).mean()

        # Compute entropy bonus
        logits = self.policy(obs_tensor)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        # Total policy loss (with entropy bonus)
        total_policy_loss = policy_loss - self.entropy_coef * entropy

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Update value network if using baseline
        if self.use_baseline:
            value_loss = F.mse_loss(self.value(obs_tensor), returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()

            self.value_losses.append(value_loss.item())

        # Store statistics
        self.policy_losses.append(policy_loss.item())
        self.entropies.append(entropy.item())

    def train(self, num_episodes: int, verbose: bool = True) -> Dict:
        """
        Train the policy for a given number of episodes.

        Args:
            num_episodes: Number of episodes to train
            verbose: Whether to print progress

        Returns:
            Dictionary of training statistics
        """
        for episode in range(num_episodes):
            # Collect episode
            obs, actions, rewards, log_probs = self.collect_episode()

            # Update policy
            self.update(obs, actions, rewards, log_probs)

            # Store episode statistics
            episode_reward = sum(rewards)
            episode_length = len(rewards)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Print progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses if self.use_baseline else [],
            "entropies": self.entropies
        }

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained policy.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []

        for _ in range(num_episodes):
            obs, actions, rewards, _ = self.collect_episode(deterministic=True)
            eval_rewards.append(sum(rewards))
            eval_lengths.append(len(rewards))

        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths)
        }

    def save(self, path: str):
        """Save model checkpoints."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
        }

        if self.use_baseline:
            checkpoint["value_state_dict"] = self.value.state_dict()
            checkpoint["value_optimizer_state_dict"] = self.value_optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoints."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        if self.use_baseline and "value_state_dict" in checkpoint:
            self.value.load_state_dict(checkpoint["value_state_dict"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])

        print(f"Model loaded from {path}")


def train_reinforce_from_config(env, config_path: str, output_dir: str, num_episodes: int = 1000):
    """
    Train REINFORCE algorithm from a configuration file.

    Args:
        env: Gymnasium environment
        config_path: Path to JSON configuration file
        output_dir: Directory to save outputs
        num_episodes: Number of episodes to train
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize algorithm
    agent = REINFORCE(
        env=env,
        learning_rate=config.get("learning_rate", 0.001),
        gamma=config.get("gamma", 0.99),
        hidden_dims=config.get("hidden_dims", [128, 128]),
        use_baseline=config.get("use_baseline", False),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        entropy_coef=config.get("entropy_coef", 0.01)
    )

    # Train
    print(f"Training REINFORCE with config: {config.get('id', 'unknown')}")
    stats = agent.train(num_episodes=num_episodes, verbose=True)

    # Save model
    model_path = os.path.join(output_dir, f"{config.get('id', 'model')}.pt")
    agent.save(model_path)

    # Save statistics
    stats_path = os.path.join(output_dir, f"{config.get('id', 'stats')}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in stats.items()}, f, indent=2)

    return agent, stats


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from environment.custom_env import ClinicEnv

    # Create environment
    env = ClinicEnv(seed=42, max_steps=500)

    # Train with default config
    agent = REINFORCE(env=env)
    stats = agent.train(num_episodes=100)

    # Evaluate
    eval_stats = agent.evaluate(num_episodes=10)
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"Mean Length: {eval_stats['mean_length']:.1f} ± {eval_stats['std_length']:.1f}")
