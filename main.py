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
from time import perf_counter
from util import generate_phi,save_coverage_snapshot


solvers.options['show_progress'] = False


def generate_pi(env_shape, action_space, n_drones):
    def pi(phi, theta, S, eps=0.9):
        sample = np.random.random()
        if sample < eps:
            # random joint action
            return {drone: np.random.choice(action_space) for drone in range(n_drones)}

        actions = list(product(*((np.arange(action_space),) * n_drones)))
        action_values = []
        for action in actions:
            A = []
            b = []
            G = []
            h = []
            c = np.zeros(n_drones * action_space)

            for i in range(n_drones):
                # c[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i])
                c[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i]).item()



            G.append(-1 * np.identity(n_drones * action_space))
            h.extend([0] * (n_drones * action_space))

            for i in range(n_drones):
                arr = np.zeros(n_drones * action_space)
                arr[i * action_space: (i + 1) * action_space] = 1
                A.append(arr)
                b.append(1)

            for i in range(n_drones):
                arr = np.zeros(n_drones * action_space)
                for a in range(action_space):
                    action_ = tuple(x if j != i else a for j, x in enumerate(action))
                    # arr[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i]) - phi(S, action_).T.dot(theta[i])
                    arr[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i]).item() - phi(S, action_).T.dot(theta[i]).item()
                                                        

                G.append(arr)
                h.append(0)

            A = matrix(np.stack(A).astype(float))
            b = matrix(np.stack(b).astype(float))
            c = matrix(np.array(c).flatten().astype(float).reshape(-1, 1))
            G = matrix(np.vstack(G).astype(float))
            h = matrix(np.array(h).astype(float).reshape(-1, 1))

            solved = solvers.lp(c, G, h, A=A, b=b)
            sol = np.array(solved['x'])
            action_values.append(np.sum([phi(S, action).T.dot(theta[i]) * sol[i * action_space + action[i]] for i in range(n_drones)]))
        action_values = np.array(action_values)
        # random tiebreaking among max value actions
        A = actions[np.random.choice(np.flatnonzero(action_values == action_values.max()))]
        return {drone: A[drone] for drone in range(n_drones)}
    return pi



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('foi', type=str, help='File containing FOI data.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('-f', action='store_true', help='Overwrite output directory if it already exists.')
    parser.add_argument('--fov', type=float, default=np.radians(30), help='Drone field of vision.')
    parser.add_argument('--env_dim', default=None, nargs=3, type=int, metavar=('X', 'Y', 'Z'), help='Environment dimensions. Will be inferred from FOI if not specified.')
    parser.add_argument('--n_drones', default=3, type=int, help='Number of drones to simulate.')
    parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--n_episodes', default=400, type=int, help='Number of episodes to simulate.')
    parser.add_argument('--episode_max_steps', default=2000, type=int, help='Maximum number of steps per episode.')
    parser.add_argument('--max_eps', default=0.95, type=float, help='Max epsilon for epsilon-greedy policy.')
    parser.add_argument('--min_eps', default=0.05, type=float, help='Min epsilon for epsilon-greedy policy.')
    parser.add_argument('--eps_decay', default=10000, type=float, help='Epsilon decay rate for epsilon-greedy policy.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--perturb_foi', default=None, nargs=2, metavar=('new_foi', 'episode'), help='Substitute original foi with new_foi at specific episode.')
    parser.add_argument('--perturb_fov', default=None, nargs=2, metavar=('new_fov', 'episode'), help='Substitute original fov with new_fov at specific episode.')
    args = parser.parse_args()

    # if os.path.exists(args.output_dir):
    #     if args.f:
    #         shutil.rmtree(args.output_dir)
    #     else:
    #         raise FileExistsError(f'Output directory {args.output_dir} already exists.')
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)

    foi = np.genfromtxt(args.foi, delimiter=',')
    env_dim = args.env_dim if args.env_dim is not None else tuple([x for x in foi.shape] + [max(foi.shape)])

    env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov, n_drones=args.n_drones)
    action_space = env.action_space.n

    # if args.n_drones <=2:
    phi, phi_dim = generate_phi(env_dim, action_space, args.n_drones)
    print(phi_dim)
    # else:
    #     num_centers = 10
    #     rbf_centers = []
    #     n_drones = 3
    #     X, Y, Z = env.shape
    #     for _ in range(num_centers):
    #         # Each drone i: (x_i,y_i,z_i)
    #         # Flatten them in order => (x0,y0,z0, x1,y1,z1, x2,y2,z2)
    #         center = []
    #         for i in range(n_drones):
    #             cx = np.random.uniform(0, X-1)  # in [0..X-1]
    #             cy = np.random.uniform(0, Y-1)  # in [0..Y-1]
    #             cz = np.random.uniform(1,  Z  ) # in [1..Z]
    #             center.extend([cx, cy, cz])
    #         rbf_centers.append(center)

    #     rbf_centers = np.array(rbf_centers)  # shape: (50, 3*n_drones)
    #     phi, phi_dim = generate_phi_rbf(env.shape,action_space,args.n_drones, rbf_centers, mu=5)
    #     print(phi_dim)

    theta = np.zeros((args.n_drones, phi_dim))

    pi = generate_pi(env_dim, action_space, args.n_drones)

    episode_rewards = np.zeros(args.n_episodes)
    episode_steps = np.zeros(args.n_episodes).astype(int)
    steps = 0
    durations = []

    try:
        for episode in range(args.n_episodes):
            state = env.reset()
            done = False
            t_start = perf_counter()
            for k in range(args.episode_max_steps):
                #### Plotting a few snapshots in the very last training episode
                if episode == args.n_episodes - 1:
                    os.makedirs(args.output_dir, exist_ok=True)
                    save_coverage_snapshot(env, k+1, args.output_dir)
                for i in range(args.n_drones):
                    epsilon = args.min_eps + (args.max_eps - args.min_eps) * math.exp(-1. * steps / args.eps_decay)
                    pi_A = pi(phi, theta, state, eps=epsilon)  
                    next_state, reward, done, meta = env.step(pi_A)

                    A = tuple([pi_A[drone] for drone in range(args.n_drones)])
                    actions = product(*((np.arange(action_space),) * args.n_drones))
                    q_next = np.max([phi(next_state, action).T.dot(theta[i]) for action in actions])
                    theta[i] = theta[i] + args.lr * (reward + args.gamma * q_next - phi(state, A).T.dot(theta[i])) * phi(state, A).flatten()
                    episode_rewards[episode] += reward
            
                    state = next_state
                    if done:
                        break
                episode_steps[episode] += 1
                steps += 1
                if done:
                    break
            tf = perf_counter()
            duration = tf-t_start
            durations.append(duration)
            print(f'Episode {episode}: {k + 1} steps, Episode reward is {episode_rewards[episode]}, duration is {duration} s')
    except KeyboardInterrupt:
        pass
    

    trained_theta_path = os.path.join(args.output_dir, "trained_theta_baseline.npy")
    np.save(trained_theta_path, theta)
    print(f"Saved trained parameters to {trained_theta_path}")

    #####################################################
    # plt.figure(dpi=150)
    # plt.ylim(0, args.episode_max_steps)
    # plt.plot(episode_steps)
    # plt.savefig(os.path.join(args.output_dir, 'episode_steps.png'))
    # pickle.dump(episode_steps, open(os.path.join(args.output_dir, 'episode_steps.pkl'), 'wb'))
    
    plt.figure(dpi=150)
    plt.plot(durations, label='Duration per episode')
    plt.xlabel('Episode')
    plt.ylabel('Durations')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'episode_durations.png'))
    print(f"Training complete. Plots and data saved in {args.output_dir}.")





if __name__ == '__main__':
    main()