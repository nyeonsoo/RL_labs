import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from gyms.simple_maze_grid import SimpleMazeGrid


def state_to_index(state, n):
    player_pos = np.argwhere(state == 1)
    if player_pos.size == 0:
        return None
    row, col = player_pos[0]
    return row * n + col

def value_iteration(env, gamma=0.99, theta=1e-6):
    V = np.zeros(env.n * env.n)
    policy = np.zeros((env.n * env.n, env.action_space.n))
    delta = float('inf')
    iteration = 0
    env.render_v_values(V, policy, iteration)

    while delta > theta:
        delta = 0
        for player_pos in env.get_all_states():
            env.set_player_pos(player_pos)
            state = env._get_state()
            state_idx = state_to_index(state, env.n)
            if state_idx is None:
                continue
            v = V[state_idx]

            # NOTE: The following can be used to visualise this algorithm's process
            if env.render_option: 
                env.render_v_values(V, policy, iteration)

            # Value iteration update: simultaneously update V(s) by finding the maximum value over actions
            q_values = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                next_state, reward, terminated = env.simulate_action(player_pos, action)[:3]
                next_state_idx = state_to_index(next_state, env.n)
                if next_state_idx is not None:
                    q_values[action] = reward + gamma * V[next_state_idx] * (not terminated)
                else:
                    q_values[action] = reward

            # Update value function for the current state
            V[state_idx] = np.max(q_values)
            delta = max(delta, abs(v - V[state_idx]))
                    
            best_action = np.argmax(q_values)
            policy[state_idx] = np.eye(env.action_space.n)[best_action]   # Set the best action as the greedy policy
        
        iteration += 1

    return policy, V, iteration


def main():
    n = 10
    k = 9
    m = 20
    random_seed = 2
    gamma = 0.9
    theta = 1e-6

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=True, random_seed=random_seed)
    policy, V, iteration = value_iteration(env, gamma=gamma, theta=theta)

    # Plot the value function
    env.render_v_values(V, policy, iteration)
    plt.imshow(V.reshape((n, n)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Value Function')
    plt.savefig('value_iteration_simple_maze_grid.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()