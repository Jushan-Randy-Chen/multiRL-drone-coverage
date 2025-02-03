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


def generate_phi_rbf(env_shape, action_space, n_drones,
                     rbf_centers,  # shape: (L, 3*n_drones) or however you encode state
                     mu):
    """
    Builds an RBF-based feature function phi(S, A).

    Args:
      env_shape: (X, Y, Z) as before
      action_space: number of discrete actions per drone
      n_drones: number of drones
      rbf_centers: a (L x state_dim) array of L "centers" in the same
                   coordinate space as your states.
      mu: the RBF bandwidth (scalar or vector).
    
    Returns:
      phi: a function phi(S,A) -> R^(L * (action_space^n_drones))
      phi_dim: the dimension of that feature vector
    """
    # 1) Precompute the dimension of the state space (in your coordinate encoding).
    #    For convenience, we treat "state_dim" as 3 * n_drones if each drone is (x,y,z).
    #    Alternatively, you might have a different flattening approach.
    state_dim = 3 * n_drones  # in your case: (x,y,z) for each drone

    # 2) Build a dictionary that maps a joint action tuple -> an integer index.
    #    Just as you did in the original fixed sparse version.
    drone_actions = np.arange(action_space)
    all_joint_actions = list(product(*(drone_actions,) * n_drones))
    actions_index_map = {tuple_a: i for i, tuple_a in enumerate(all_joint_actions)}

    # 3) Define a helper to flatten the environment state S into shape = (state_dim,).
    #    For instance, if each of the n_drones has a position (x,y,z),
    #    we concatenate them in order: (x1,y1,z1, x2,y2,z2, ..., xN,yN,zN).
    def _flatten_state(S):
        # S is a list (or array) of shape (n_drones, 3).
        flat = []
        for i in range(n_drones):
            x, y, z = S[i]
            flat.extend([x, y, z])
        return np.array(flat, dtype=float)  # shape (3*n_drones,)

    # 4) The RBF feature for a single state S (flattened) wrt a single center c.
    #    phi_RBF = exp(- ||S - c||^2 / (2 * mu^2))
    #    We'll define a function that returns the L-dimensional RBF vector.
    def _rbf_features(flat_state):
        # flat_state: shape (state_dim,)
        # rbf_centers: shape (L, state_dim)
        diffs = flat_state - rbf_centers  # shape (L, state_dim)
        # Squared Euclidean distance from S to each center.
        sqdist = np.sum(diffs**2, axis=1)  # shape (L,)
        # Compute RBF value: e^(-sqdist / (2 * mu^2))
        rbf_vals = np.exp(-sqdist / (2 * (mu**2)))
        return rbf_vals  # shape (L,)

    # 5) Define phi(S,A):
    #    - Flatten the state to shape (state_dim,)
    #    - Compute the L-dimensional RBF vector for this state
    #    - Build a zero vector of length L * (action_space^n_drones)
    #    - Fill in the slot that corresponds to action A with the L RBF values
    phi_dim = len(rbf_centers) * (action_space ** n_drones)
    def phi(S, A):
        # Flatten the environment state
        flat_state = _flatten_state(S)  # shape (3*n_drones,)
        # Compute the RBF features for the state
        rbf_vals = _rbf_features(flat_state)  # shape (L,)
        
        # Build big zero vector
        feat = np.zeros(phi_dim, dtype=float)
        
        # Which "slot" do we fill?  The index for the chosen action:
        a_idx = actions_index_map[A]
        
        # The offset where we place the L RBF values
        offset = a_idx * len(rbf_vals)
        
        # Fill
        feat[offset: offset + len(rbf_vals)] = rbf_vals
        return feat.reshape(-1, 1)  # shape: (phi_dim,1) just like your fixed-sparse version

    return phi, phi_dim
