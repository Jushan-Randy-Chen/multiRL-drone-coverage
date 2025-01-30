import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

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


def save_coverage_snapshot(env, step_idx, output_dir):
    """
    Example function that makes a 2D plot of the drones' coverage FOV
    and the FOI points, similar to your example figure.
    """
    plt.figure(figsize=(5,5))
    # Plot FOI points or region
    foi = env.foi
    X, Y = foi.shape
    # If your environment has a known bounding shape, plot it
    # e.g. show the discrete points as '*' or something
    # (You can adapt as needed to replicate your figure exactly.)

    foi_points = np.argwhere(foi>0)  # indices where foi is > 0
    plt.scatter(foi_points[:,1], foi_points[:,0], marker='*', color='red')
    
    # print(f"DEBUG: Number of drones found = {len(env._drones)}")

    # Plot each drone’s bounding box for FOV
    for i, drone in env._drones.items():
        x, y, z = drone.pos
        # print(x,y,z)
        # print(f"DEBUG: Drone {i} position: (x={x}, y={y}, z={z})")

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