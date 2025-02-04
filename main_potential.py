import argparse
import numpy as np
import os
import shutil
import math
import random
import pickle
import matplotlib.pyplot as plt
from itertools import product
from field_coverage_env import FieldCoverageEnv
from potential_game_env import generate_phi, PotentialGameEnv
import argparse
from util import save_coverage_snapshot, generate_phi_rbf
from time import perf_counter

def q_learning_potential_fa(args, env, phi_func, actions_dict,
                            n_episodes=500,
                            max_steps=2000,
                            gamma=0.9,
                            alpha=0.1,
                            eps_max=0.95,
                            eps_min=0.05,
                            eps_decay=1e5):
    """
    Runs single-agent Q-learning on PotentialGameEnv with the linear Q-function:
      Q(s,a) = theta^T phi_func(s,a).

    - env: the PotentialGameEnv wrapper around FieldCoverageEnv
    - phi_func: your generate_phi(...) function returning phi(S, A)
    - actions_dict: dict mapping (a1,...,aN) -> integer index, used to decode.
    - n_episodes, max_steps: training loops
    - gamma, alpha: standard Q-learning hyperparameters
    - eps_{max,min,decay}: Epsilon-greedy schedule

    Returns:
      theta: learned parameter vector
      episode_rewards: list of total potential-based reward per episode
    """
    # 1) Build reverse mapping: int -> (a1,...,aN)
    #    so we can decode an action index into a joint-action tuple
    all_actions = sorted(actions_dict.keys(), key=lambda a: actions_dict[a])

    # 2) Initialize theta
    #    We can do a quick call to phi_func on a dummy state+action
    #    to get the dimension.  E.g., use the environment reset state
    #    plus the first action.
    init_state = env.reset()
    init_tuple = all_actions[0]  # e.g. (0,0,...)

    if args.n_drones <=2:
        phi_sample = phi_func(init_state, init_tuple)
        phi_dim = phi_sample.size
    else:
        num_centers = 6
        rbf_centers = []
        n_drones = 3
        X, Y, Z = env.env.shape
        for _ in range(num_centers):
            # Each drone i: (x_i,y_i,z_i)
            # Flatten them in order => (x0,y0,z0, x1,y1,z1, x2,y2,z2)
            center = []
            for i in range(n_drones):
                cx = np.random.uniform(0, X-1)  # in [0..X-1]
                cy = np.random.uniform(0, Y-1)  # in [0..Y-1]
                cz = np.random.uniform(1,  Z  ) # in [1..Z]
                center.extend([cx, cy, cz])
            rbf_centers.append(center)

        rbf_centers = np.array(rbf_centers)  # shape: (50, 3*n_drones)
        phi, phi_dim = generate_phi_rbf(env.env.shape, 
                                        env.action_space.n,
                                        args.n_drones, 
                                        rbf_centers, 
                                        mu=5)

    # reset env again to start fresh
    env.reset()

    theta = np.zeros(phi_dim, dtype=np.float64)
    convergence_threshold = 0.001
    theta_prev = theta.copy()
    # 3) Epsilon decay function
    steps_done = 0
    def get_epsilon(t):
        return eps_min + (eps_max - eps_min) * math.exp(-1.0 * t / eps_decay)

    episode_rewards = []
    durations = []
    # 4) Main training loop
    rel_diff = np.inf
    # converge_ep = np.inf
    for ep in range(n_episodes):
        obs = env.reset()  # shape (n_drones,3)
        ep_reward = 0.0 
        t_start  = perf_counter()
        
        for t in range(max_steps):
            if ep == n_episodes - 1:
                os.makedirs(args.output_dir, exist_ok=True)
                save_coverage_snapshot(env, t+1, args.output_dir)
            # if ep == converge_ep + 1:
            #     save_coverage_snapshot(env, t+1, args.output_dir)
       
            epsilon = get_epsilon(steps_done)

            # Compute Q-values for all possible actions
            qvals = []
            for action_idx in range(env.action_space.n):
                A_tuple = all_actions[action_idx]
                phi_sa = phi_func(obs, A_tuple)[:, 0]  # shape (phi_dim,)
                qvals.append(theta.dot(phi_sa))

            # Epsilon-greedy
            if random.random() < epsilon:
                action_idx = np.random.randint(env.action_space.n)
            else:
                action_idx = np.argmax(qvals)

            # Step environment
            next_obs, reward, done, info = env.step(action_idx)
            ep_reward += reward #>>> Our immediate reward here is the potential difference J(s')-J(s)
            
            # Next Q max
            next_qvals = []
            for action_idx2 in range(env.action_space.n):
                A_tuple2 = all_actions[action_idx2]
                phi_sprime_a = phi_func(next_obs, A_tuple2)[:, 0]
                next_qvals.append(theta.dot(phi_sprime_a))
            max_next_q = np.max(next_qvals)

            # TD error
            chosen_tuple = all_actions[action_idx]
            phi_sa_chosen = phi_func(obs, chosen_tuple)[:, 0]
            current_q = theta.dot(phi_sa_chosen)
            td_error = reward + gamma * max_next_q - current_q

            # Update
            theta += alpha * td_error * phi_sa_chosen

            obs = next_obs  
            steps_done += 1
            if done:
                break

        tf = perf_counter()
        duration = tf-t_start
        durations.append(duration)

        episode_rewards.append(ep_reward)

        print(f"[Ep {ep}] steps:{t+1}  sumReward:{ep_reward:.2f}  eps:{epsilon:.3f}  duration for episode:{duration:.2f}")

        rel_diff = np.linalg.norm(theta_prev - theta) / np.linalg.norm(theta)
        print(f'Theta difference is {rel_diff}')

        theta_prev = theta.copy()


    return theta, episode_rewards, durations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('foi', type=str, help='File containing FOI data.')
    parser.add_argument('output_dir', type=str, help='Output directory.')
    parser.add_argument('-f', action='store_true', help='Overwrite output directory if it already exists.')
    parser.add_argument('--fov', type=float, default=np.radians(30), help='Drone field of vision.')
    parser.add_argument('--env_dim', default=None, nargs=3, type=int,
                        metavar=('X', 'Y', 'Z'),
                        help='Environment dimensions. Inferred from FOI if not specified.')
    parser.add_argument('--n_drones', default=3, type=int, help='Number of drones to simulate.')
    parser.add_argument('--gamma', default=0.9, type=float, help='Discount factor.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--n_episodes', default=500, type=int, help='Number of episodes to simulate.')
    parser.add_argument('--episode_max_steps', default=2000, type=int, help='Maximum number of steps per episode.')
    parser.add_argument('--max_eps', default=0.95, type=float, help='Max epsilon for epsilon-greedy policy.')
    parser.add_argument('--min_eps', default=0.05, type=float, help='Min epsilon for epsilon-greedy policy.')
    parser.add_argument('--eps_decay', default=10000, type=float, help='Epsilon decay rate.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--perturb_foi', default=None, nargs=2,
                        metavar=('new_foi', 'episode'),
                        help='Substitute original FOI with new_foi at specific episode.')
    parser.add_argument('--perturb_fov', default=None, nargs=2,
                        metavar=('new_fov', 'episode'),
                        help='Substitute original FOV with new_fov at specific episode.')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Handle output directory
    # if os.path.exists(args.output_dir):
    #     if args.f:
    #         shutil.rmtree(args.output_dir)
    #     else:
    #         raise FileExistsError(f"Output directory {args.output_dir} already exists. Use -f to overwrite.")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load FOI from CSV
    foi = np.genfromtxt(args.foi, delimiter=',')
    if args.env_dim is not None:
        env_dim = tuple(args.env_dim)
    else:
        # Guess environment shape from FOI
        # If FOI is shape (X, Y), pick Z=some integer or use max(X,Y)
        X, Y = foi.shape
        Z = max(X, Y)  # or pick some specific value
        env_dim = (X, Y, Z)

    print(f"Environment dimension: {env_dim}")

    # 2) Create the multi-drone coverage environment
    base_env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov,
                                n_drones=args.n_drones,
                                max_steps=args.episode_max_steps,
                                seed=args.seed)

    # 3) Wrap it into single-agent environment
    single_env = PotentialGameEnv(base_env)

    # 4) Generate phi(s,a) from your function
    phi_func, phi_dim = generate_phi(env_dim, 6, args.n_drones)  # 6 actions per drone

    # 5) Build the dictionary of (joint_action) -> index
    all_joint = list(product(range(6), repeat=args.n_drones))
    actions_dict = {joint: i for i, joint in enumerate(all_joint)}

    # 6) Train with single-agent Q-learning + linear function approximation
    theta, episode_rewards, durations = q_learning_potential_fa(args,
                                                    env=single_env,
                                                    phi_func=phi_func,
                                                    actions_dict=actions_dict,
                                                    n_episodes=args.n_episodes,
                                                    max_steps=args.episode_max_steps,
                                                    gamma=args.gamma,
                                                    alpha=args.lr,
                                                    eps_max=args.max_eps,
                                                    eps_min=args.min_eps,
                                                    eps_decay=args.eps_decay
                                                )

    # 7) Save results
    np.save(os.path.join(args.output_dir, "theta_potential_game.npy"), theta)
    pickle.dump(episode_rewards, open(os.path.join(args.output_dir, "episode_rewards.pkl"), 'wb'))

    # 8) Plot
    plt.figure(dpi=150)
    plt.plot(episode_rewards, label='Episode Total Reward (incremental potential)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'episode_rewards.png'))

    plt.figure(dpi=150)
    plt.plot(durations, label='Duration per episode')
    plt.xlabel('Episode')
    plt.ylabel('Durations')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'episode_durations.png'))
    print(f"Training complete. Plots and data saved in {args.output_dir}.")


if __name__ == '__main__':
    main()