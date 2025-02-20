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

def epsilon_greedy_policy(Q, state_idx, epsilon, num_actions):
    """ε-greedy policy: choose a random action with probability ε, otherwise the greedy action"""
    if np.random.rand() < epsilon:
        return np.random.choice(num_actions)  # Explore (random action)
    else:
        return np.argmax(Q[state_idx])  # Exploit (best action)

def generate_episode(env, policy, trajectory_length, epsilon, Q, episode_index):

    # Reset the environment
    state, _ = env.retry()

    trajectory = []
    state_action_pairs = []

    for _ in range(trajectory_length):  # Limiting the trajectory length
        state_idx = state_to_index(state, env.n)
        if state_idx is None:  # The player is at either the pit or goal
            break

        # Select an action using the ε-greedy policy
        action = epsilon_greedy_policy(Q, state_idx, epsilon, env.action_space.n)

        next_state, reward, terminated, _ = env.step(action)
        trajectory.append((state_idx, action, reward))
        state_action_pairs.append((state_idx, action))

        if env.render_option:
            env.render_q_values(Q, episode_index, with_arrow=True)

        state = next_state

    return trajectory, state_action_pairs


def monte_carlo_e_soft(env, gamma, num_episodes, trajectory_length, epsilon):
    policy = np.ones((env.n * env.n, env.action_space.n)) / env.action_space.n
    Q = np.zeros((env.n * env.n, env.action_space.n))
    returns = {state: {action: [] for action in range(env.action_space.n)} for state in range(env.n * env.n)}

    for episode_index in range(num_episodes):
        # Generate an episode with ε-soft action selection
        trajectory, visited_state_action_pairs = generate_episode(env, policy, trajectory_length, epsilon, Q, episode_index)

        G = 0  # Initialize return value

        for t in reversed(range(len(trajectory))):
            state_idx, action, reward = trajectory[t]
            G = gamma * G + reward  # Accumulate discounted rewards

            # If state-action pair is first visited in the episode
            if (state_idx, action) not in visited_state_action_pairs[:t]:
                # Append G to the list of returns for (state, action)
                returns[state_idx][action].append(G)

                # Update Q(s, a) with the average of returns for this (state, action) pair
                Q[state_idx][action] = np.mean(returns[state_idx][action])

                # Improve the policy to be ε-soft with respect to Q
                best_action = np.argmax(Q[state_idx])
                for a in range(env.action_space.n):
                    if a == best_action:
                        policy[state_idx][a] = 1 - epsilon + (epsilon / env.action_space.n)
                    else:
                        policy[state_idx][a] = epsilon / env.action_space.n
    
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
    epsilon = 0.1
    num_episodes = 1000
    trajectory_length = n * 5

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=True, random_seed=random_seed)
    policy, Q = monte_carlo_e_soft(env, gamma=gamma, num_episodes=num_episodes, trajectory_length=trajectory_length, epsilon=epsilon)

    # Plot the policy or Q-values
    env.render_q_values(Q, num_episodes, with_arrow=True)
    plt.imshow(np.max(Q, axis=1).reshape((n, n)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('State-Action Values (Q)')
    plt.savefig('monte_carlo_e_soft_simple_maze_grid.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()
