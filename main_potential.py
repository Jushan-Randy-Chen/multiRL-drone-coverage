import argparse
import numpy as np
from field_coverage_env import FieldCoverageEnv
from itertools import product, combinations
from cvxopt import solvers, matrix
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import pickle
import shutil
import os
import time

solvers.options['show_progress'] = False


def generate_phi(env_shape, action_space, n_drones):
    X, Y, Z = env_shape
    state_dim = (X + Y + Z) * n_drones

    drone_actions = np.arange(action_space)
    actions = {x: i for i, x in enumerate(product(*((drone_actions,) * n_drones)))}
    def phi(S, A):
        states = []
        for i in range(n_drones):
            x, y, z = S[i]
            arr = np.zeros(X)
            arr[x] = 1
            states.append(arr)

            arr = np.zeros(Y)
            arr[y] = 1
            states.append(arr)

            arr = np.zeros(Z)
            arr[z] = 1
            states.append(arr)
        states = np.concatenate(states)
        state = np.zeros(len(states) * (action_space ** n_drones))
        action_slot = actions[A] * len(states)
        state[action_slot: action_slot + len(states)] = states
        return state.astype(int).reshape(-1, 1)
    return phi, state_dim * action_space ** n_drones


def compute_potential(phi, theta, S, A, n_drones):
    """
    Computes the global potential function J(S, A) as the sum of all agents' Q-values. We realize that we have a Markov Potential Game
    """
    return sum([phi(S, A).T.dot(theta[i]).item() for i in range(n_drones)])

def generate_pi(env_shape, action_space, n_drones):
    def pi(phi, theta, S, eps=0.9):
        if np.random.random() < eps:
            return {drone: np.random.choice(action_space) for drone in range(n_drones)}
        
        # Evaluate all joint actions and select the one maximizing the sum of Q-values
        joint_actions = list(product(*((np.arange(action_space),) * n_drones)))
        best_value = -np.inf
        best_action = None
        
        for action in joint_actions:
            # Sum of individual Q-values for this joint action
            total_q = sum(phi(S, action).T.dot(theta[i]).item() for i in range(n_drones))
            if total_q > best_value:
                best_value = total_q
                best_action = action
                
        return {drone: best_action[drone] for drone in range(n_drones)}
    return pi



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('foi', type=str, help='File containing FOI data.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('-f', action='store_true', help='Overwrite output directory if it already exists.')
    parser.add_argument('--fov', type=float, default=np.radians(30), help='Drone field of vision.')
    parser.add_argument('--env_dim', default=None, nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='Environment dimensions.')
    parser.add_argument('--n_drones', default=3, type=int, help='Number of drones.')
    parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--n_episodes', default=500, type=int, help='Training episodes.')
    parser.add_argument('--episode_max_steps', default=2000, type=int, help='Max steps per episode.')
    parser.add_argument('--max_eps', default=0.95, type=float, help='Max exploration rate.')
    parser.add_argument('--min_eps', default=0.05, type=float, help='Min exploration rate.')
    parser.add_argument('--eps_decay', default=10000, type=float, help='Epsilon decay rate.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--perturb_foi', default=None, nargs=2, metavar=('new_foi', 'episode'), help='FOI perturbation.')
    parser.add_argument('--perturb_fov', default=None, nargs=2, metavar=('new_fov', 'episode'), help='FOV perturbation.')
    args = parser.parse_args()

    # Initialize environment and directories
    if os.path.exists(args.output_dir) and args.f:
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load FOI and initialize environment
    foi = np.genfromtxt(args.foi, delimiter=',')
    env_dim = args.env_dim or (foi.shape[0], foi.shape[1], max(foi.shape))
    env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov, n_drones=args.n_drones)
    
    # Initialize learning components
    phi, phi_dim = generate_phi(env_dim, env.action_space.n, args.n_drones)
    theta = np.zeros((args.n_drones, phi_dim))
    pi = generate_pi(env_dim, env.action_space.n, args.n_drones)

    # Training metrics
    episode_rewards = np.zeros(args.n_episodes)
    episode_steps = np.zeros(args.n_episodes, dtype=int)
    coverage_history = []
    steps = 0

    try:
        for episode in range(args.n_episodes):
            # Handle perturbations
            if args.perturb_foi and episode == int(args.perturb_foi[1]):
                env.foi = np.genfromtxt(args.perturb_foi[0], delimiter=',')
            if args.perturb_fov and episode == int(args.perturb_fov[1]):
                env.fov = float(args.perturb_fov[0])

            state = env.reset()
            episode_coverage = []

            for k in range(args.episode_max_steps):
                # Select joint action using potential-based policy
                epsilon = args.min_eps + (args.max_eps - args.min_eps) * math.exp(-steps/args.eps_decay)
                pi_A = pi(phi, theta, state, eps=epsilon)
                
                # Environment step
                next_state, individual_rewards, global_reward, done, meta = env.step(pi_A)
                episode_coverage.append(meta['coverage'])
                
                # Q-learning updates
                A = tuple(pi_A[drone] for drone in range(args.n_drones))
                for i in range(args.n_drones):
                    # Calculate target using potential-based reward
                    actions = product(*[range(env.action_space.n)]*args.n_drones)
                    q_next = max(phi(next_state, a).T.dot(theta[i]) for a in actions)
                    td_error = individual_rewards[i] + args.gamma*q_next - phi(state, A).T.dot(theta[i])
                    theta[i] += args.lr * td_error * phi(state, A).flatten()

                # Update tracking variables
                state = next_state
                episode_rewards[episode] += global_reward
                episode_steps[episode] = k+1
                steps += 1
                if done: break

            # Record coverage progress
            coverage_history.append(np.mean(episode_coverage))
            print(f"Episode {episode}: {episode_steps[episode]} steps, Final Coverage {episode_coverage[-1]:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Save results
    plt.figure(dpi=150)
    plt.plot(coverage_history)
    plt.xlabel("Episode")
    plt.ylabel("Average Coverage")
    plt.savefig(os.path.join(args.output_dir, 'coverage_progress.png'))
    pickle.dump(coverage_history, open(os.path.join(args.output_dir, 'coverage.pkl'), 'wb'))

if __name__ == '__main__':
    main()
