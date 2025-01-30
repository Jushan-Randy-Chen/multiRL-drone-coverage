import argparse
import numpy as np
from field_coverage_env import FieldCoverageEnv
from itertools import product
import math
import matplotlib.pyplot as plt
import pickle
import shutil
import os

class MPGAgent:
    def __init__(self, phi_dim, action_space, lr, gamma):
        self.theta = np.zeros(phi_dim)
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma

    def update(self, phi, phi_next, reward):
        td_error = reward + self.gamma * np.max(phi_next) - phi.dot(self.theta)
        self.theta += self.lr * td_error * phi

def generate_phi(env_shape, n_drones):
    """Simplified state representation for potential game formulations"""
    X, Y, Z = env_shape
    # Feature vector: one-hot encoding of each drone's x,y,z coordinates
    return lambda S: np.concatenate([
        np.eye(X)[x] for x, y, z in S] + [
        np.eye(Y)[y] for x, y, z in S] + [
        np.eye(Z)[z] for x, y, z in S
    ]), (X + Y + Z) * n_drones

def generate_policy(action_space, n_drones, phi):
    def pi(agents, S, eps):
        if np.random.random() < eps:
            return {drone: np.random.choice(action_space) for drone in range(n_drones)}
        
        # Decentralized action selection based on current estimates
        return {
            drone: np.argmax([
                agents[drone].theta.dot(phi(S)) 
                for a in range(action_space)
            ])
            for drone in range(n_drones)
        }
    return pi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('foi', type=str, help='File containing FOI data.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('-f', action='store_true', help='Overwrite output directory if exists.')
    parser.add_argument('--fov', type=float, default=np.radians(30), help='Drone field of vision.')
    parser.add_argument('--env_dim', default=None, nargs=3, type=int, 
                      metavar=('X', 'Y', 'Z'), help='Environment dimensions.')
    parser.add_argument('--n_drones', default=1, type=int, help='Number of drones.')
    parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--n_episodes', default=1000, type=int, help='Number of episodes.')
    parser.add_argument('--episode_max_steps', default=20000, type=int, help='Max steps per episode.')
    parser.add_argument('--max_eps', default=0.9, type=float, help='Max exploration rate.')
    parser.add_argument('--min_eps', default=0.05, type=float, help='Min exploration rate.')
    parser.add_argument('--eps_decay', default=10000, type=float, help='Epsilon decay rate.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    args = parser.parse_args()

    # Environment setup
    if os.path.exists(args.output_dir):
        if args.f: shutil.rmtree(args.output_dir)
        else: raise FileExistsError(args.output_dir)
    os.makedirs(args.output_dir)
    np.random.seed(args.seed)

    foi = np.genfromtxt(args.foi, delimiter=',')
    env_dim = args.env_dim if args.env_dim else (foi.shape[0], foi.shape[1], 10)
    
    env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov, n_drones=args.n_drones)
    phi, phi_dim = generate_phi(env_dim, args.n_drones)
    
    # Initialize agents
    agents = [MPGAgent(phi_dim, env.action_space.n, args.lr, args.gamma) 
             for _ in range(args.n_drones)]

    pi = generate_policy(env.action_space.n, args.n_drones, phi)
    
    # Training metrics
    episode_rewards = np.zeros(args.n_episodes)
    episode_potentials = np.zeros(args.n_episodes)
    steps = 0

    try:
        for episode in range(args.n_episodes):
            state = env.reset()
            done = False
    
            for k in range(args.episode_max_steps):
                # Calculate exploration rate
                epsilon = args.min_eps + (args.max_eps - args.min_eps) * math.exp(-steps/args.eps_decay)
                
                # Get joint action
                action = pi(agents, state, epsilon)
                
                # take step in environment
                next_state, individual_rewards, global_reward, done, info = env.step(action, potential=True)

                # Update each agent
                for i, agent in enumerate(agents):
                    # Calculate individual reward (marginal contribution)
                    reward_i = individual_rewards[i]
                    episode_potentials[episode] += reward_i
                    # Get feature vectors
                    phi_s = phi(state)
                    phi_s_next = phi(next_state)
                    
                    # Update agent parameters
                    agent.update(phi_s, phi_s_next, reward_i)
                
                state = next_state
                steps += 1
                if done: break

            # Record episode metrics
            episode_rewards[episode] = global_reward
            print(f"Episode {episode}: {k+1} steps, Global reward {episode_rewards[episode]:.1f}, Cumulative Individual Rewards {episode_potentials[episode]:.1f},Success {info['success']}")

    except KeyboardInterrupt:
        pass

    # Save trained parameters at the end of training
    theta_values = np.array([agent.theta for agent in agents])  # Store all agent parameters
    theta_path = os.path.join(args.output_dir, "trained_theta.npy")
    np.save(theta_path, theta_values)
    print(f"Saved trained theta parameters to {theta_path}")

    # Save results
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episode_rewards)
    plt.title('Global Success Rate')
    plt.subplot(1,2,2)
    plt.plot(episode_potentials)
    plt.title('Average Potential')
    plt.savefig(os.path.join(args.output_dir, 'results.png'))
    
    pickle.dump({
        'rewards': episode_rewards,
        'potentials': episode_potentials
    }, open(os.path.join(args.output_dir, 'train_data.pkl'), 'wb'))

if __name__ == '__main__':
    main()