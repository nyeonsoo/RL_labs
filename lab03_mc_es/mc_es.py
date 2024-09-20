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

def generate_episode_with_exploring_starts(env, policy, trajectory_length, Q, episode_index):

    # Reset environment with new initial player position
    state, _ = env.retry()
    all_states = env.get_all_states()            
    random_player_pos = all_states[np.random.choice(len(all_states))]
    env.set_player_pos(random_player_pos)


    # Generate an episode with exploring starts
    trajectory = []
    state_action_pairs = []

    state = env._get_state()
    for _ in range(trajectory_length):  # Limiting the trajectory length
        state_idx = state_to_index(state, env.n)
        if state_idx is None: # the player is at either pit or goal. 
            break

        # Record the state-action pair and its reward
        if len(trajectory) == 0: # If this is the first state-action pair
            # Select a random action for exploring starts (even if policy exists)
            action = np.random.choice(env.action_space.n)    
        else:            
            # Select an action probabilistically based on the policy
            action = np.random.choice(env.action_space.n, p=policy[state_idx])            


        next_state, reward, terminated, _ = env.step(action)
        trajectory.append((state_idx, action, reward))
        state_action_pairs.append((state_idx, action))

        if env.render_option:
            env.render_q_values(Q, episode_index, with_arrow=True)

        state = next_state

    return trajectory, state_action_pairs


def monte_carlo_es(env, gamma=0.99, num_episodes=10000, trajectory_length = 100):
    policy = np.ones((env.n * env.n, env.action_space.n)) / env.action_space.n
    Q = np.zeros((env.n * env.n, env.action_space.n))
    returns = {state: {action: [] for action in range(env.action_space.n)} for state in range(env.n * env.n)}

    for episode_index in range(num_episodes):

        # Generate an episode with exploring starts (random initial state and action)
        trajectory, visited_state_action_pairs = generate_episode_with_exploring_starts(env, policy,  trajectory_length, Q, episode_index)

        G = 0        

        for t in reversed(range(len(trajectory))):

            state_idx, action, reward = trajectory[t]
            # TODO: Implement the Monte Carlo ES algorithm


    return policy, Q

def main():
    # For (1)
    n = 5
    k = 3
    m = 4
    random_seed = 42

    # For (2)
    # n = 10
    # k = 9
    # m = 20
    # random_seed = 2

    gamma = 0.9
    num_episodes = 1000
    trajectory_length = n*5

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=True, random_seed=random_seed)
    policy, Q = monte_carlo_es(env, gamma=gamma, num_episodes=num_episodes, trajectory_length=trajectory_length)

    # Plot the policy or Q-values
    env.render_q_values(Q, num_episodes, with_arrow=True)
    plt.imshow(np.max(Q, axis=1).reshape((n, n)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('State-Action Values (Q)')
    plt.savefig('monte_carlo_es_simple_maze_grid.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()
