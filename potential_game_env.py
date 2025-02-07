import numpy as np
from itertools import product, combinations
import gym
from math import factorial

class PotentialGameEnv(gym.Env):
    """
    Single-agent wrapper around FieldCoverageEnv that:
      - has a discrete action space of size (6^n_drones),
      - returns a scalar reward = incremental change in coverage potential
        from one step to the next.
    """

    def __init__(self, multiagent_env):
        super().__init__()
        self.env = multiagent_env
        self.foi = self.env.foi  # Access the FOI data from the base environment
        self._drones = self.env._drones  # Inherit drones from the base environment
        self.n_drones = multiagent_env.n_drones
        self.single_drone_action_size = 6  # from FieldCoverageEnv.Action
        
        # All possible joint actions (a Cartesian product)
        self.joint_actions = list(product(range(self.single_drone_action_size), 
                                          repeat=self.n_drones))
        self.action_space = gym.spaces.Discrete(len(self.joint_actions))
        

        # Observation: a list of n_drones positions = shape (n_drones, 3)
        # We'll represent it in a Box that spans the environment's shape
        X, Y, Z = self.env.shape
        high = np.array([X, Y, Z], dtype=np.float32)
        high = np.tile(high, (self.n_drones,1))
        # self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.array(self.env.shape), dtype=np.float32)

        # Track last potential so we can give the incremental reward
        self._last_potential = None
        self._steps = 0

    def reset(self):
        obs = self.env.reset()  # list of (x, y, z) for each drone
        self._drones = self.env._drones
        self._steps = 0
        self._last_potential = self._compute_potential()
        return np.array(obs, dtype=np.float32)

    def step(self, single_action_index):
        # 1) Decode the single-agent action into a dictionary for each drone
        joint_action_tuple = self.joint_actions[single_action_index]
        action_dict = {i: joint_action_tuple[i] for i in range(self.n_drones)}
        
        # 2) Step the underlying multi-drone environment
        #    (We'll ignore its built-in 'reward' or 'global_reward'—instead we use potential.)
        next_obs_list, _, _, info = self.env.step(action_dict)
        
        # 3) Compute the new potential, and define reward = Δ potential !!!!
        # new_potential = self._compute_potential()
        # new_potential = self._compute_potential_2()
        new_potential = self._compute_potential_sigmoid() 
        # reward = new_potential - self._last_potential

        reward = new_potential
        done = reward >= 0.099
        # self._last_potential = new_potential
        self._steps += 1
        
        # 4) Convert next_obs_list to a NumPy array
        next_obs = np.array(next_obs_list, dtype=np.float32)
        return next_obs, reward, done, info

    def _compute_potential(self):
        """
        Compute the global potential (coverage) using the inclusion–exclusion principle.
        For a set S of drones, let φ_S be the area where the FOI and each drone in S have True.
        Then:
        J = ∑_{k=1}^{n_drones} ∑_{S ⊆ {1,…,n_drones}, |S|=k} [(-1)^(k-1) * (k-1)! * φ_S]
        """
        n_drones = self.n_drones
        masks = self.env._view_masks()
        # Ensure FOI is boolean
        foi = self.env.foi.astype(bool)
        
        potential = 0
        for k in range(1, n_drones + 1):
            for subset in combinations(range(n_drones), k):
                inter = foi.copy()
                for i in subset:
                    # Using bitwise AND here, assuming masks[i] is boolean
                    inter = inter & masks[i]
                phi_S = np.sum(inter)
                coefficient = ((-1)**(k-1)) * factorial(k-1)
                potential += coefficient * phi_S

        return potential
    
    def _compute_potential_2(self):
        masks = self.env._view_masks()
        foi = self.env.foi.astype(bool)
        coverage = 0
        for i, drone in self._drones.items():
            coverage += np.sum(masks[i].flatten() & foi.flatten())
        
        drones = set(self._drones.keys(),)
        overlap = 0
        if len(drones) > 1:
            for i, drone in self._drones.items():
                mask = masks[i]
                other_masks = np.sum([masks[x] for x in drones - {i}], axis=0)
                overlap += np.sum(mask.flatten() & other_masks.flatten())
  
        potential = coverage - overlap    

        return potential
    
    def _compute_potential_sigmoid(self):
        masks = self.env._view_masks()
        foi = self.env.foi.astype(bool)
        coverage = 0
        for i, drone in self._drones.items():
            coverage += np.sum(masks[i].flatten() & foi.flatten())
        
        drones = set(self._drones.keys(),)
        overlap = 0
        if len(drones) > 1:
            for i, drone in self._drones.items():
                mask = masks[i]
                other_masks = np.sum([masks[x] for x in drones - {i}], axis=0)
                overlap += np.sum(mask.flatten() & other_masks.flatten())
  
        H = coverage - overlap

        global_reward = 0.1/(1+np.exp(-H))

        return global_reward


def generate_phi(env_shape, action_space, n_drones):
    """
    Returns:
      phi(S,A): a one-hot vector for (state, joint_action).
      phi_dim: dimension of that vector = (state_dim) * (action_space^n_drones).

    Specifically, state_dim = (X + Y + Z)*n_drones
    and action_space^n_drones enumerates all possible joint actions.

    So phi(S,A) is a one-hot vector that first encodes the state,
    then places it at an offset corresponding to the index of the joint action A.
    """
    X, Y, Z = env_shape
    state_dim = (X + Y + Z) * n_drones

    # Create a dictionary that maps a tuple A=(a1,a2,...,an) to an integer index
    drone_actions = np.arange(action_space)
    # e.g. actions[(0,0,...,0)] = 0, actions[(0,0,...,1)] = 1, ...
    all_joint_actions = list(product(*(drone_actions,)*n_drones))
    actions = {tuple_a: i for i, tuple_a in enumerate(all_joint_actions)}

    def phi(S, A):
        """
        S: list of (x,y,z) for each drone, or array shape (n_drones, 3).
        A: tuple of length n_drones, each an action in [0..action_space-1].

        Returns a (phi_dim, 1) one-hot vector. The "state part" is repeated
        in the slice for the chosen joint action.
        """
        # Build one-hot for the state
        # For each drone i, we do one-hot for x, y, z.
        states = []
        for i in range(n_drones):
            x, y, z = S[i]
            x, y, z = int(x), int(y), int(z)

            # one-hot for x dimension
            arr_x = np.zeros(X)
            arr_x[x] = 1
            states.append(arr_x)

            # one-hot for y dimension
            arr_y = np.zeros(Y)
            arr_y[y] = 1
            states.append(arr_y)

            # one-hot for z dimension
            arr_z = np.zeros(Z)
            arr_z[z] = 1
            states.append(arr_z)

        # Concatenate all drone one-hots into shape = (state_dim,)
        states_vec = np.concatenate(states, axis=0)

        # Build a final big vector of length state_dim * action_space^n_drones
        phi_vec = np.zeros(len(states_vec) * (action_space**n_drones), dtype=int)

        # The offset for the chosen joint action
        action_offset = actions[A] * len(states_vec)
        # Place the state vector in that offset
        phi_vec[action_offset: action_offset + len(states_vec)] = states_vec
        
        return phi_vec.reshape(-1, 1)

    # dimension of that big vector
    phi_dim = state_dim * (action_space ** n_drones)
    return phi, phi_dim