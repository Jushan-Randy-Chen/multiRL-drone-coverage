import numpy as np
from scipy.optimize import minimize

class SOVAAgent:
    def __init__(self, agent_id, position, state_indices, action_indices, theta_star, rho=1.0):
        """
        agent_id: Unique identifier
        position: Agent's position (for sensing radius checks)
        state_indices: Global indices of the agent's local state subset
        action_indices: Global indices of the agent's local action subset
        theta_star: Trained Q-function parameters (phi^T * theta_star = Q)
        rho: ADMM penalty parameter
        """
        self.id = agent_id
        self.position = position
        self.state_indices = state_indices  # Local state subset (global indices)
        self.action_indices = action_indices  # Local action subset (global indices)
        self.theta = theta_star.copy()  # Local copy of trained parameters
        self.rho = rho
        
        # Local variables (state and action)
        self.x = np.zeros(len(state_indices))  # Local state vector
        self.a = np.zeros(len(action_indices))  # Local action vector
        
        # Dual variables for consensus constraints
        self.lambda_x = np.zeros(len(state_indices))  # State consensus dual
        self.lambda_a = np.zeros(len(action_indices))  # Action consensus dual
        
        self.neighbors = []  # List of neighbor agents

    def get_overlap(self, neighbor, variable_type='state'):
        """
        Return overlapping indices between this agent and a neighbor.
        variable_type: 'state' or 'action'
        """
        if variable_type == 'state':
            self_vars = self.state_indices
            neighbor_vars = neighbor.state_indices
        else:
            self_vars = self.action_indices
            neighbor_vars = neighbor.action_indices
        
        overlap_global = list(set(self_vars).intersection(neighbor_vars))
        overlap_self = [self_vars.index(g) for g in overlap_global]
        overlap_neighbor = [neighbor_vars.index(g) for g in overlap_global]
        return overlap_self, overlap_neighbor
    
def update_neighbors(agents, sensing_radius=3.0):
    """Update neighbors for all agents based on their positions."""
    for agent in agents:
        agent.neighbors = []
        for other in agents:
            if agent != other and np.linalg.norm(agent.position - other.position) <= sensing_radius:
                agent.neighbors.append(other)


def feature_map(agent, global_state, global_action):
    """
    Construct phi(S, A) for the agent using its local subset of S and A.
    global_state: Full state vector (not observed by the agent)
    global_action: Full action vector (not observed by the agent)
    """
    # Extract local state and action from global vectors
    local_state = global_state[agent.state_indices]
    local_action = global_action[agent.action_indices]
    
    # Example: Concatenate state and action (modify as needed)
    phi = np.concatenate([local_state, local_action])
    return phi.reshape(-1, 1)  # Column vector

def primal_update(agent, global_state, global_action, neighbors_x, neighbors_a):
    """Solve for x_i^{k+1} and a_i^{k+1} using ADMM consensus terms."""
    def objective(local_vars):
        # Split variables into state (x) and action (a)
        x_new = local_vars[:len(agent.state_indices)]
        a_new = local_vars[len(agent.state_indices):]
        
        # Update global state/action with local variables
        global_state_new = global_state.copy()
        global_action_new = global_action.copy()
        global_state_new[agent.state_indices] = x_new
        global_action_new[agent.action_indices] = a_new
        
        # Compute Q-value: phi(S, A)^T * theta
        phi = feature_map(agent, global_state_new, global_action_new)
        Q = phi.T @ agent.theta
        
        # Consensus penalty terms (state and action)
        penalty_x = 0.0
        penalty_a = 0.0
        for neighbor in agent.neighbors:
            # State overlap penalty
            overlap_x_self, overlap_x_neighbor = agent.get_overlap(neighbor, 'state')
            x_neighbor = neighbors_x[neighbor.id][overlap_x_neighbor]
            diff_x = x_new[overlap_x_self] - x_neighbor
            penalty_x += np.dot(diff_x, diff_x)
            
            # Action overlap penalty
            overlap_a_self, overlap_a_neighbor = agent.get_overlap(neighbor, 'action')
            a_neighbor = neighbors_a[neighbor.id][overlap_a_neighbor]
            diff_a = a_new[overlap_a_self] - a_neighbor
            penalty_a += np.dot(diff_a, diff_a)
        
        return -Q.item() + 0.5 * agent.rho * (penalty_x + penalty_a)
    
    # Solve local optimization (BFGS for continuous variables)
    initial_guess = np.concatenate([agent.x, agent.a])
    result = minimize(objective, initial_guess, method='BFGS')
    x_new, a_new = np.split(result.x, [len(agent.state_indices)])
    return x_new, a_new


def dual_update(agent, x_new, a_new, neighbors_x_new, neighbors_a_new):
    """Update lambda_x and lambda_a using new neighbor values."""
    delta_lambda_x = np.zeros_like(agent.lambda_x)
    delta_lambda_a = np.zeros_like(agent.lambda_a)
    
    for neighbor in agent.neighbors:
        # State dual update
        overlap_x_self, overlap_x_neighbor = agent.get_overlap(neighbor, 'state')
        x_neighbor_new = neighbors_x_new[neighbor.id][overlap_x_neighbor]
        delta_lambda_x[overlap_x_self] += x_new[overlap_x_self] - x_neighbor_new
        
        # Action dual update
        overlap_a_self, overlap_a_neighbor = agent.get_overlap(neighbor, 'action')
        a_neighbor_new = neighbors_a_new[neighbor.id][overlap_a_neighbor]
        delta_lambda_a[overlap_a_self] += a_new[overlap_a_self] - a_neighbor_new
    
    agent.lambda_x += agent.rho * delta_lambda_x
    agent.lambda_a += agent.rho * delta_lambda_a


def simulate_sova_admm(agents, global_state, global_action, max_iters=50):
    """Run SOVA-ADMM for decentralized execution."""
    for _ in range(max_iters):
        # Update neighbors based on sensing radius
        update_neighbors(agents)
        
        # Exchange current local variables with neighbors
        neighbors_x = {agent.id: agent.x for agent in agents}
        neighbors_a = {agent.id: agent.a for agent in agents}
        
        # Perform primal update for all agents
        new_xs = {}
        new_as = {}
        for agent in agents:
            x_new, a_new = primal_update(agent, global_state, global_action, neighbors_x, neighbors_a)
            new_xs[agent.id] = x_new
            new_as[agent.id] = a_new
        
        # Exchange new variables for dual update
        neighbors_x_new = {agent.id: new_xs[agent.id] for agent in agents}
        neighbors_a_new = {agent.id: new_as[agent.id] for agent in agents}
        
        # Perform dual update for all agents
        for agent in agents:
            dual_update(agent, new_xs[agent.id], new_as[agent.id], neighbors_x_new, neighbors_a_new)
            agent.x = new_xs[agent.id]
            agent.a = new_as[agent.id]
    
    return agents


# Global state and action space (example: 5 states, 3 actions)
global_state = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
global_action = np.array([0, 1, 2])

# Initialize agents with overlapping subsets
agents = [
    SOVAAgent(
        agent_id=0,
        position=np.array([1.0, 2.0]),
        state_indices=[0, 1, 2],  # Controls states 0, 1, 2
        action_indices=[0, 1],     # Controls actions 0, 1
        theta_star=np.random.randn(5),  # Trained parameters (phi size = 5)
        rho=1.0
    ),
    SOVAAgent(
        agent_id=1,
        position=np.array([3.0, 4.0]),
        state_indices=[2, 3, 4],  # Controls states 2, 3, 4
        action_indices=[1, 2],     # Controls actions 1, 2
        theta_star=np.random.randn(5),
        rho=1.0
    )
]

# Simulate SOVA-ADMM consensus
agents = simulate_sova_admm(agents, global_state, global_action, max_iters=50)

# Check consensus on overlapping variables
print("Agent 0 State:", agents[0].x)
print("Agent 1 State:", agents[1].x)
print("Agent 0 Action:", agents[0].a)
print("Agent 1 Action:", agents[1].a)