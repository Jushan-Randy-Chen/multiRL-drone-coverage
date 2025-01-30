import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from field_coverage_env import FieldCoverageEnv
from itertools import product
from util import generate_phi 

# 1) A decentralized policy that picks each drone’s action independently.
def decentralized_policy(phi, theta, state, n_drones, action_space):
    """
    For each drone i, pick a_i = argmax_{a_i} Q_i(state, a_i).
    This requires you to define how to evaluate Q_i without enumerating all drones' actions.
    """
    actions_dict = {}
    for i in range(n_drones): #looping over each drone
        best_action = None #initialize
        best_value = -float('inf')

        # Explore all possible actions for drone i
        for a_i in range(action_space):
            # One simple approach: fix other drones' actions to 0 or do not factor them in.
            # We'll build a "joint" action where drone i = a_i, others = 0
            dummy_joint = [0]*n_drones
            dummy_joint[i] = a_i
            q_val = phi(state, tuple(dummy_joint)).T.dot(theta[i]).item()
            if q_val > best_value:
                best_value = q_val
                best_action = a_i

        actions_dict[i] = best_action
    return actions_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theta_path', type=str, required=True,
                        help='Path to the trained theta .npy file')
    parser.add_argument('--foi', type=str, required=True,
                        help='Path to FOI data (same as used in training)')
    parser.add_argument('--fov', type=float, default=np.radians(30),
                        help='Drone field of vision (same as training)')
    parser.add_argument('--env_dim', default=None, nargs=3, type=int,
                        metavar=('X','Y','Z'),
                        help='Environment dimensions, must match training if not None')
    parser.add_argument('--n_drones', default=2, type=int,
                        help='Number of drones (same as training)')
    parser.add_argument('--episode_max_steps', default=2000, type=int,
                        help='Number of steps in the evaluation episode')
    parser.add_argument('--snapshot_steps', default='1,10,20,30',
                        help='Comma-separated list of steps to save a coverage snapshot')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Where to save the snapshot plots')
    args = parser.parse_args()

    # 1) Loading trained parameters
    theta = np.load(args.theta_path, allow_pickle=True) 
    print(f"Loaded theta of shape {theta.shape}")

    # 2) Create the environment
    foi = np.genfromtxt(args.foi, delimiter=',')
    if args.env_dim is not None:
        env_dim = tuple(args.env_dim)
    else:
        env_dim = (foi.shape[0], foi.shape[1], max(foi.shape))
    env = FieldCoverageEnv(env_dim, foi=foi, fov=args.fov, n_drones=args.n_drones)

    # 3) the same phi(...) function used in training:
    phi, _ = generate_phi(env_dim, env.action_space.n, args.n_drones)

    # Prepare snapshot steps
    snapshot_steps = [int(x) for x in args.snapshot_steps.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)

    # 4) Run ONE episode (or more if you like) with the decentralized policy
    state = env.reset()
    done = False

    for k in range(args.episode_max_steps):
        # Let each drone pick its action independently
        pi_A = decentralized_policy(phi, theta, state, args.n_drones, env.action_space.n)
        
        # Step the environment
        next_state, individual_rewards, global_reward, done, meta = env.step(pi_A, potential=True)

        # Check if we want a snapshot at this step
        if (k+1) in snapshot_steps:
            save_coverage_snapshot(env, k+1, args.output_dir)

        if done:
            break
        state = next_state


def save_coverage_snapshot(env, step_idx, output_dir):
    """
    Example function that makes a 2D plot of the drones' coverage FOV
    and the FOI points, similar to your example figure.
    """
    import matplotlib.patches as patches

    plt.figure(figsize=(5,5))
    # Plot FOI points or region
    foi = env.foi
    X, Y = foi.shape
    # If your environment has a known bounding shape, plot it
    # e.g. show the discrete points as '*' or something
    # (You can adapt as needed to replicate your figure exactly.)

    foi_points = np.argwhere(foi>0)  # indices where foi is > 0
    plt.scatter(foi_points[:,1], foi_points[:,0], marker='*', color='red')

    # Plot each drone’s bounding box for FOV
    for i, drone in env._drones.items():
        x, y, z = drone.pos

        # Simple “rectangle” approximation:
        # FOV in x-direction = tan(fov) * z
        # FOV in y-direction = tan(fov) * z
        x_range = np.tan(drone.fov) * z
        y_range = np.tan(drone.fov) * z

        # For a 2D top-down plot, let's just make a rectangle:
        # Lower-left corner:
        x_min = x - x_range
        y_min = y - y_range
        rect_width = 2 * x_range
        rect_height = 2 * y_range

        rect = patches.Rectangle((y_min, x_min),  # note: (col, row) reversed if needed
                                 rect_height, rect_width,
                                 linewidth=2, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title(f"Coverage at step k={step_idx}")
    plt.xlim([0, Y])  # x-axis is columns
    plt.ylim([0, X])  # y-axis is rows
    plt.gca().invert_yaxis()  # if you want (0,0) at bottom-left
    plt.grid(True)
    outpath = os.path.join(output_dir, f"coverage_step_{step_idx}.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved snapshot figure {outpath}")


if __name__ == '__main__':
    main()
