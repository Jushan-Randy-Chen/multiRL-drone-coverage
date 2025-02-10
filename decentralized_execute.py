#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Import your environment classes.
from field_coverage_env import FieldCoverageEnv
from potential_game_env import PotentialGameEnv
from main_potential_DQN import DQN
from util import save_coverage_snapshot
import matplotlib.pyplot as plt


#########################################
# Helper functions
#########################################
def joint_action_to_onehot(joint_action, n_agents, num_actions_per_agent):
    """
    Convert a joint action (list of agent actions) to a one-hot vector.
    We assume a lexicographic ordering where agent 0 is the most significant digit.
    
    Args:
        joint_action: List of integers (length n_agents).
        n_agents: Number of agents.
        num_actions_per_agent: Number of actions per agent.
    
    Returns:
        one_hot: A numpy array of shape (num_actions_per_agent**n_agents,) with one 1.
    """
    joint_index = 0
    for a in joint_action:
        joint_index = joint_index * num_actions_per_agent + a
    global_action_space = num_actions_per_agent ** n_agents
    one_hot = np.zeros(global_action_space, dtype=np.float32)
    one_hot[joint_index] = 1.0
    return one_hot

def joint_action_to_index(joint_action, num_actions_per_agent):
    """
    Convert a joint action (list of agent actions) into a single integer index.
    """
    joint_index = 0
    for a in joint_action:
        joint_index = joint_index * num_actions_per_agent + a
    return joint_index

def evaluate_q(model, state, joint_action_onehot, device):
    """
    Evaluate the Q value from the model for the given state and joint action.
    
    Args:
        model: The loaded DQN model.
        state: A numpy array of shape (state_dim,).
        joint_action_onehot: A numpy array of shape (global_action_space,).
        device: torch.device.
    
    Returns:
        q_val: The scalar Q value.
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # (1, state_dim)
    action_tensor = torch.FloatTensor(joint_action_onehot).unsqueeze(0).to(device)  # (1, action_dim)
    with torch.no_grad():
        q_val = model(state_tensor, action_tensor)
    return q_val.item()

def decentralized_best_response(state, initial_joint_action, eval_q_func,
                                n_agents, n_iterations, num_actions_per_agent):
    """
    Perform best-response updates for each agent in a potential game.
    
    Args:
        state: Global state (as a flat numpy array).
        initial_joint_action: A list of actions (one per agent).
        eval_q_func: Function that takes (state, one_hot_joint_action) and returns a scalar Q value.
        n_agents: Number of agents.
        n_iterations: Number of rounds of best-response updates.
        num_actions_per_agent: Discrete action space size for each agent.
        
    Returns:
        joint_action: The updated joint action as a list.
    """
    joint_action = initial_joint_action.copy()
    for _ in range(n_iterations):
        # Agents update sequentially.
        for i in range(n_agents):
            best_action = None
            best_q = -float('inf')
            for candidate_action in range(num_actions_per_agent):
                candidate_joint_action = joint_action.copy()
                candidate_joint_action[i] = candidate_action
                one_hot = joint_action_to_onehot(candidate_joint_action, n_agents, num_actions_per_agent)
                q_value = eval_q_func(state, one_hot)
                if q_value > best_q:
                    best_q = q_value
                    best_action = candidate_action
            joint_action[i] = best_action
    return joint_action

#########################################
# Main execution script
#########################################
def main():
    parser = argparse.ArgumentParser(description="Decentralized Execution using Best-Response")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained Q network .pth file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to saving result')
    parser.add_argument('--foi_csv', type=str, required=True,
                        help='CSV file containing the Field-of-Interest')
    parser.add_argument('--env_dim', type=int, nargs=3,
                        help='Environment dimensions: X Y Z')
    parser.add_argument('--fov', type=float, default=np.radians(30),
                        help='Field-of-view (in degrees; will be converted to radians)')
    parser.add_argument('--n_agents', type=int, default=2,
                        help='Number of agents/drones')
    parser.add_argument('--num_actions_per_agent', type=int, default=6,
                        help='Number of discrete actions per agent')
    parser.add_argument('--n_iterations', type=int, default=5,
                        help='Number of best-response iterations per time step')
    parser.add_argument('--n_episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=20,
                        help='Maximum number of steps per episode')
    args = parser.parse_args()

    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the Field-of-Interest from CSV.
    foi = np.genfromtxt(args.foi_csv, delimiter=',')
    
    # Create the base environment and wrap it if needed.
    base_env = FieldCoverageEnv(shape=tuple(args.env_dim),
                                foi=foi,
                                fov=np.radians(args.fov),
                                n_drones=args.n_agents,
                                max_steps=args.max_steps)
    env = PotentialGameEnv(base_env)
    
    # Define the global action space dimension.
    global_action_space = args.num_actions_per_agent ** args.n_agents
    
    # Assume state is a flattened vector of drone positions: state_dim = n_agents * 3.
    state_dim = args.n_agents * 3
    action_dim = global_action_space

    # Initialize the DQN model and load the trained weights.
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded trained model from", args.model_path)
    
    # Wrap the evaluate_q function to include the model and device.
    def eval_q(state, joint_action_onehot):
        return evaluate_q(model, state, joint_action_onehot, device)
    
    # For initial joint action, we choose a default (e.g., all zeros).
    initial_joint_action = [0] * args.n_agents

    eps_rewards = []

    # Run the execution loop.
    for ep in range(args.n_episodes):
        obs = env.reset()  # Expecting a joint observation, e.g., list of drone positions.
        joint_action = initial_joint_action.copy()
        total_reward = 0.0
        done = False
        step = 0
        print(f"\nEpisode {ep} starting...")
        while not done and step < args.max_steps:
            if ep == args.n_episodes-1:
                if step % 5 == 0:
                    save_coverage_snapshot(env, step, args.output_dir)

            # The global state is assumed to be the flattened positions.
            state = np.array(obs).flatten()  # shape: (state_dim,)
            
            # Compute the joint action via decentralized best response.
            joint_action = decentralized_best_response(state, joint_action,
                                                       eval_q, args.n_agents,
                                                       args.n_iterations,
                                                       args.num_actions_per_agent)
            # print(f'Joint action {joint_action}')
            
            # Convert the joint action list to a single integer index.
            joint_action_index = joint_action_to_index(joint_action, args.num_actions_per_agent)
            # print(f'joint action index {joint_action_index}')
            
            # Step the environment.
            obs, reward, done, info = env.step(joint_action_index)
            total_reward += reward
            eps_rewards.append(total_reward)
            step += 1
            
            # Optionally, print per-step information.
            print(f"Step {step}: Joint Action: {joint_action} | Reward: {reward:.2f}")
            
        print(f"Episode {ep} finished in {step} steps with total reward: {total_reward:.2f}")


    ### save plots
    plt.figure(dpi=150)
    plt.plot(eps_rewards, label='Episodic Rewards')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig(f'{args.output_dir}/rewards.png')

if __name__ == '__main__':
    main()
