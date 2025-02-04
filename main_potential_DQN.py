#!/usr/bin/env python3
import argparse
import os
import math
import random
import pickle
import numpy as np
from itertools import product
from time import perf_counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import your environment classes. For example:
from field_coverage_env import FieldCoverageEnv
from potential_game_env import PotentialGameEnv
from util import save_coverage_snapshot  # Assuming you have this function

# -------------------------
# Define the DQN Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
    parser.add_argument('--n_episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--episode_max_steps', type=int, default=2000, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size for training')
    parser.add_argument('--buffer_capacity', type=int, default=10000, help='Replay buffer capacity')
    parser.add_argument('--eps_start', type=float, default=0.95, help='Initial epsilon for epsilon-greedy')
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

    # Define the state dimension.
    # Here we assume the state is the positions of the drones: shape (n_drones, 3)
    # We flatten this to get state_dim = 3 * n_drones.
    state_dim = args.n_drones * 3

    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}')

    # Initialize the policy network and the target network.
    policy_net = DQN(state_dim, num_actions).to(device)
    target_net = DQN(state_dim, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.buffer_capacity)

    steps_done = 0

    def get_epsilon(step):
        return args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1. * step / args.eps_decay)

    def select_action(state, eps_threshold):
        # Convert the state (shape: (n_drones, 3)) to a flat tensor.
        state_flat = np.array(state).flatten()  # shape: (state_dim,)
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(device)
        if random.random() < eps_threshold:
            return random.randrange(num_actions)
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                return q_values.max(1)[1].item()

    episode_rewards = []
    durations = []

    for ep in range(args.n_episodes):
        obs = env.reset()  # Observation is a list (or array) of drone positions, shape (n_drones, 3)
        ep_reward = 0.0
        t_start = perf_counter()

        for t in range(args.episode_max_steps):
            if ep == args.n_episodes - 1:
                save_coverage_snapshot(env,t+1,args.output_dir)

            epsilon = get_epsilon(steps_done)
            action_idx = select_action(obs, epsilon)
            next_obs, reward, done, info = env.step(action_idx)
            ep_reward += reward

            # Store the transition in the replay buffer.
            replay_buffer.push(np.array(obs).flatten(), action_idx, reward,
                               np.array(next_obs).flatten(), done)
            obs = next_obs
            steps_done += 1

            # Perform a learning step if enough samples are available.
            if len(replay_buffer) >= args.batch_size:
                transitions = replay_buffer.sample(args.batch_size)
                batch = list(zip(*transitions))
                state_batch = torch.FloatTensor(np.stack(batch[0])).to(device)
                action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
                reward_batch = torch.FloatTensor(batch[2]).to(device)
                next_state_batch = torch.FloatTensor(np.stack(batch[3])).to(device)
                done_batch = torch.FloatTensor(batch[4]).to(device)

                # Compute Q(s, a) using policy network.
                current_q = policy_net(state_batch).gather(1, action_batch).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_state_batch).max(1)[0]
                target_q = reward_batch + args.gamma * next_q * (1 - done_batch)

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

        # Update the target network every 'target_update' episodes.
        if ep % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    trained_model_path = os.path.join(args.output_dir, "trained_dqn.pth")
    torch.save(policy_net.state_dict(), trained_model_path)
    print(f"Trained model saved to {trained_model_path}")

    # Save training logs.
    with open(os.path.join(args.output_dir, "episode_rewards.pkl"), "wb") as f:
        pickle.dump(episode_rewards, f)
    with open(os.path.join(args.output_dir, "durations.pkl"), "wb") as f:
        pickle.dump(durations, f)

if __name__ == "__main__":
    main()
