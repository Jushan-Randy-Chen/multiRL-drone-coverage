#!/usr/bin/env python3
import argparse
import os
import math
import random
import pickle
import numpy as np
from itertools import product
from time import perf_counter
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import your environment classes. For example:
from field_coverage_env import FieldCoverageEnv
from potential_game_env import PotentialGameEnv
from util import save_coverage_snapshot, plot_coverage_masks  # Assuming you have this function

# -------------------------
# Define the DQN Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Args:
          state_dim: Dimension of the state (e.g., 3*n_drones for positions).
          action_dim: Dimension for representing the action. For discrete actions,
                      you can use a one-hot encoding so that action_dim equals the number
                      of discrete actions.
          hidden_dim: Number of hidden units.
        """
        super(DQN, self).__init__()
        # The network takes as input the concatenation of state and action.
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output a single Q-value

    def forward(self, state, action):
        """
        Forward pass for a state-action pair.

        Args:
          state: A tensor of shape (batch_size, state_dim)
          action: A tensor of shape (batch_size, action_dim)
        Returns:
          A tensor of shape (batch_size, 1) representing Q(s,a).
        """
        # Concatenate state and action along the feature dimension.
        x = torch.cat([state, action], dim=1) #shape :(action_dim,state_dim + action_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# -------------------------
# Define a simple Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# -------------------------
# Main Training Function
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foi', type=str, required=True,
                        help='Path to CSV file containing the Field-Of-Interest')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saving results and snapshots')
    parser.add_argument('--n_drones', type=int, default=3, help='Number of drones')
    parser.add_argument('--fov', type=float, default=np.radians(30), help='Drone field of view (radians)')
    parser.add_argument('--env_dim', type=int, nargs=3, default=None,
                        help='Environment dimensions: X Y Z. If not provided, derived from FOI.')
    parser.add_argument('--n_episodes', type=int, default=400, help='Number of training episodes')
    parser.add_argument('--episode_max_steps', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for training')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='Replay buffer capacity')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Final epsilon')
    parser.add_argument('--eps_decay', type=int, default=1e5, help='Epsilon decay rate')
    parser.add_argument('--target_update', type=int, default=10, help='Frequency (in episodes) to update target network')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load FOI from CSV.
    foi = np.genfromtxt(args.foi, delimiter=',')
    if args.env_dim is not None:
        env_dim = tuple(args.env_dim)
    else:
        X, Y = foi.shape
        Z = max(X, Y)
        env_dim = (X, Y, Z)
    print("Environment dimensions:", env_dim)

    # Create the base multi-agent environment and wrap it.
    base_env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov,
                                 n_drones=args.n_drones,
                                 max_steps=args.episode_max_steps)
    env = PotentialGameEnv(base_env)

    # The joint action space is precomputed by the wrapper.
    num_actions = env.action_space.n  # e.g., for 3 drones with 6 actions each: 6^3 = 216
    print(f'Dimension of action space is {num_actions}')

    # For the new DQN, we use a critic that takes state and action as input.
    # We represent actions using a one-hot encoding. Thus, action_dim equals num_actions.
    action_dim = num_actions

    # Define the state dimension.
    # We assume the state is the positions of the drones: shape (n_drones, 3)
    # We flatten this to get state_dim = 3 * n_drones.
    state_dim = args.n_drones * 3

    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}')

    # Initialize the policy network and the target network.
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Count total parameters in the policy network:
    total_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print("Total parameters in the policy network:", total_params)

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.buffer_capacity)

    steps_done = 0

    # Precompute candidate one-hot action vectors.
    candidate_actions = torch.eye(num_actions, device=device)  # shape: (num_actions, num_actions)

    def get_epsilon(step):
        return args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * step / args.eps_decay)

    def select_action(state, eps_threshold):
        """
        Select an action using epsilon-greedy over Q(s, a) values computed for all candidate actions.
        """
        # Convert state (shape: (n_drones, 3)) to a flat tensor.
        state_flat = np.array(state).flatten()  # shape: (state_dim,)
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)  # shape: (1, state_dim)
        if random.random() < eps_threshold:
            return random.randrange(num_actions)
        else:
            # For all candidate actions (one-hot vectors), compute Q(s, a).
            # Repeat the state tensor to match the number of candidate actions.
            state_repeated = state_tensor.repeat(num_actions, 1)  # shape: (num_actions, state_dim)
            q_values = policy_net(state_repeated, candidate_actions)  # shape: (num_actions, 1)
            q_values = q_values.squeeze(1)  # shape: (num_actions,)
            return q_values.argmax().item()

    episode_rewards = []
    durations = []
    steps_to_plot = [10, 20, 50, 100, 150]

    # Hyperparameters for early stopping based on Q–value improvement
    q_improve_tol = 0.01     # tolerance threshold for improvement
    patience = 10            # number of consecutive episodes to wait before stopping

    # Initialize variables before the training loop
    prev_avg_q = None
    patience_counter = 0
    training_converged = False

    for ep in tqdm(range(args.n_episodes)):
        obs = env.reset()  # Observation: list/array of drone positions, shape (n_drones, 3)
        ep_reward = 0.0
        t_start = perf_counter()
        for t in range(args.episode_max_steps):
            if (training_converged) or (ep==args.n_episodes - 1):
                if (t+1) in steps_to_plot:
                    # plot_coverage_masks(env, t + 1, args.output_dir)
                    save_coverage_snapshot(env, t+1, args.output_dir)

            epsilon = get_epsilon(steps_done)
            action_idx = select_action(obs, epsilon)
            next_obs, reward, done, info = env.step(action_idx)
            ep_reward += reward

            # Store transition in the replay buffer.
            replay_buffer.push(np.array(obs).flatten(), action_idx, reward,
                               np.array(next_obs).flatten(), done)
            obs = next_obs
            steps_done += 1

            # Perform a learning step if enough samples are available.
            if len(replay_buffer) >= args.batch_size:
                transitions = replay_buffer.sample(args.batch_size)
                batch = list(zip(*transitions))
                state_batch = torch.FloatTensor(np.stack(batch[0])).to(device)  # shape: (batch_size, state_dim)
                # Convert action indices to one-hot vectors.
                action_batch = F.one_hot(torch.LongTensor(batch[1]), num_classes=num_actions).float().to(device)  # (batch_size, action_dim)
                reward_batch = torch.FloatTensor(batch[2]).to(device)
                next_state_batch = torch.FloatTensor(np.stack(batch[3])).to(device)
                done_batch = torch.FloatTensor(batch[4]).to(device)

                # Compute current Q-values for the taken actions.
                current_q = policy_net(state_batch, action_batch).squeeze(1)  # shape: (batch_size,)

                # Compute target Q-value by taking max over candidate actions for each next state.
                batch_size_val = next_state_batch.size(0)
                # Expand next_state_batch to evaluate all candidate actions.
                # next_state_exp: (batch_size, num_actions, state_dim)
                next_state_exp = next_state_batch.unsqueeze(1).repeat(1, num_actions, 1)
                # candidate_actions is (num_actions, action_dim), expand to (batch_size, num_actions, action_dim)
                candidate_actions_exp = candidate_actions.unsqueeze(0).repeat(batch_size_val, 1, 1)
                # Flatten: shape (batch_size*num_actions, state_dim) and (batch_size*num_actions, action_dim)
                next_state_flat = next_state_exp.view(-1, state_dim)
                candidate_actions_flat = candidate_actions_exp.view(-1, num_actions)
                next_q_flat = target_net(next_state_flat, candidate_actions_flat)  # (batch_size*num_actions, 1)
                next_q_flat = next_q_flat.view(batch_size_val, num_actions)
                max_next_q = next_q_flat.max(dim=1)[0]

                target_q = reward_batch + args.gamma * max_next_q * (1 - done_batch)
                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        tf = perf_counter()
        duration = tf - t_start
        durations.append(duration)
        episode_rewards.append(ep_reward)
        print(f"[Ep {ep}] Steps: {t+1}, Episode Reward: {ep_reward:.2f}, Eps: {epsilon:.3f}, Duration: {duration:.2f}s")

        # Evaluate the current Q–function on a set of states (e.g., from the replay buffer or fixed states)
        # Here, we'll sample a small batch from the replay buffer if possible.
        if len(replay_buffer) >= args.batch_size:
            sample_transitions = replay_buffer.sample(args.batch_size)
            sample_states = torch.FloatTensor(np.stack([s for s,_,_,_,_ in sample_transitions])).to(device)
            # We need to compute Q(s,a) for all candidate actions and take the maximum for each state.
            batch_size_val = sample_states.size(0)
            # candidate_actions is of shape (num_actions, action_dim)
            candidate_actions_exp = candidate_actions.unsqueeze(0).repeat(batch_size_val, 1, 1)
            sample_states_exp = sample_states.unsqueeze(1).repeat(1, num_actions, 1)
            # Flatten the batch to get all state-action pairs.
            sample_states_flat = sample_states_exp.view(-1, state_dim)
            candidate_actions_flat = candidate_actions_exp.view(-1, num_actions)
            # Compute Q for each state-action pair using the target network
            with torch.no_grad():
                q_vals_flat = target_net(sample_states_flat, candidate_actions_flat)  # shape: (batch_size_val*num_actions, 1)
            q_vals = q_vals_flat.view(batch_size_val, num_actions)
            # For each state, take the maximum Q value
            avg_q = q_vals.max(dim=1)[0].mean().item()
        else:
            # Fallback if not enough samples
            avg_q = 0.0

        print(f"Average Q-value at episode {ep}: {avg_q:.6f}")
        
        # Check if Q-value has improved compared to the previous episode.
        if prev_avg_q is not None:
            if abs(avg_q - prev_avg_q) < q_improve_tol:
                patience_counter += 1
                print(f"Q improvement below threshold for {patience_counter} episodes.")
            else:
                patience_counter = 0

            if patience_counter >= patience:
                print(f"Convergence threshold met at episode {ep}.")
                training_converged = True  

        else:
            patience_counter = 0

        prev_avg_q = avg_q

        # Update the target network periodically.
        if ep % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())


    trained_model_path = os.path.join(args.output_dir, "trained_dqn.pth")
    torch.save(policy_net.state_dict(), trained_model_path)
    print(f"Trained model saved to {trained_model_path}")

    plt.figure(dpi=150)
    plt.plot(durations, label='Duration per episode')
    plt.xlabel('Episode')
    plt.ylabel('Durations')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'episode_durations.png'))
    print(f"Training complete. Plots and data saved in {args.output_dir}.")

    # Save training logs.
    with open(os.path.join(args.output_dir, "episode_rewards.pkl"), "wb") as f:
        pickle.dump(episode_rewards, f)
    with open(os.path.join(args.output_dir, "durations.pkl"), "wb") as f:
        pickle.dump(durations, f)

    

if __name__ == "__main__":
    main()