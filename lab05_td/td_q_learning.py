import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
from gyms.simple_maze_grid import SimpleMazeGrid

def state_to_index(state, n):
    player_pos = np.argwhere(state == 1)
    if player_pos.size == 0:  # Check if player_pos is empty
        return 0
    row, col = player_pos[0]
    return row * n + col

def choose_action(state_index, Q, epsilon, action_space):
    if np.random.rand() < epsilon:
        return action_space.sample()  # Explore
    else:
        return np.argmax(Q[state_index])  # Exploit
    
def q_table_learning(env, num_episodes=100, alpha=0.1, gamma=0.99, epsilon=0.1, render_option=False):
    Q = np.zeros((env.n * env.n, env.action_space.n))
    total_rewards = []
    for episode_index in range(num_episodes):
        # Initialize S
        state, _ = env.retry()
        state_index = state_to_index(state, env.n)
        total_reward = 0
        terminated = False

        # Loop for each step of episode
        while not terminated:

            # Choose action A from S
            action = choose_action(state_index, Q, epsilon, env.action_space)
            
            # Take action A and observe R, S'
            next_state, reward, terminated, _ = env.step(action)
            next_state_index = state_to_index(next_state, env.n)

            # TODO: Implement Q-learning
            # Q-learning update rule 
            Q[state_index, action] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action])

            # Update S to S'
            state_index = next_state_index

            # Etc
            total_reward += reward

            if render_option:
                env.render_q_values(Q, episode_index, with_arrow=True)                

        total_rewards.append(total_reward)

    return Q, total_rewards

def main():
    num_runs = 20 # NOTE: This is just to investigate the learning curve
    render_option = False

    num_episodes = 1000

    # Q-learning parameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.2  # Exploration-exploitation tradeoff

    # Initialize environment
    n = 10  # Grid size

    all_rewards = np.zeros((num_runs, num_episodes))
 
    for run in range(num_runs):
        # env = SimpleMazeGrid(n=n, k=k, m=m, render_option=render_option, random_seed=random_seed)
        player_pos = [n-1, 0]
        goal_pos = [n-1, n-1]
        pits = [[n-1, i] for i in range(1, n-1)]
        spec = player_pos, goal_pos, pits
        env = SimpleMazeGrid(n=n, render_option=render_option, spec=spec)

        Q, total_rewards = q_table_learning(env, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon, render_option=render_option)
        all_rewards[run] = total_rewards
        if run != num_runs - 1:
            env.close() 

    average_rewards = np.mean(all_rewards, axis=0)

    # Plot the average rewards over all runs
    if render_option:
        env.render_q_values(Q, num_episodes, with_arrow=True)
    plt.plot(average_rewards, label=f'$\\alpha$ = {alpha}; $\\epsilon$ = {epsilon}')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.ylim(-200, 100)    
    plt.title(f'Q-Table Learning Training (Average Over {num_runs} Runs)')    
    plt.grid(True)
    plt.legend()
    plt.savefig('results/q_table_learning_simple_maze_grid_average.png')
    plt.show()
    
    
    print("Completed!")

if __name__ == "__main__":
    main()
