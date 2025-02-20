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

def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(env.n * env.n)
    delta = float('inf')
    
    while delta > theta:
        delta = 0
        for player_pos in env.get_all_states():
            env.set_player_pos(player_pos)
            state = env._get_state()
            state_idx = state_to_index(state, env.n)
            if state_idx is None:
                continue
            v = V[state_idx]

            # TODO: Implement the policy evaluation algorithm
            new_v = 0

            # 정책에 따라 각 행동에 대해 가치 계산
            for action, action_prob in enumerate(policy[state_idx]):
                next_state, reward, terminated = env.simulate_action(player_pos, action)[:3]
                next_state_idx = state_to_index(next_state, env.n)
                if next_state_idx is not None:
                    new_v += action_prob * (reward + gamma * V[next_state_idx] * (not terminated))
                else:
                    new_v += action_prob * reward
            
            # 가치 함수 업데이트
            V[state_idx] = new_v
            delta = max(delta, abs(v - V[state_idx]))

    return V

def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros((env.n * env.n, env.action_space.n))
    for player_pos in env.get_all_states():
        env.set_player_pos(player_pos)
        state = env._get_state()
        state_idx = state_to_index(state, env.n)
        if state_idx is None:
            continue        
        # TODO: Implement the policy improvement algorithm        
        q_values = np.zeros(env.action_space.n)
        
        # 각 행동에 대해 Q(s, a) 계산
        for action in range(env.action_space.n):
            next_state, reward, terminated = env.simulate_action(player_pos, action)[:3]
            next_state_idx = state_to_index(next_state, env.n)
            if next_state_idx is not None:
                q_values[action] = reward + gamma * V[next_state_idx] * (not terminated)
            else:
                q_values[action] = reward
        
        # Q 값이 가장 큰 행동을 선택하여 정책을 업데이트
        best_action = np.argmax(q_values)
        policy[state_idx][best_action] = 1.0  # Greedy 선택
    
    return policy

def policy_iteration(env, gamma=0.99, theta=1e-6):
    policy = np.ones((env.n * env.n, env.action_space.n)) / env.action_space.n  # Initial policy (uniform random)
    iteration = 0
    
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        
        if np.all(policy == new_policy):
            break
        
        policy = new_policy
        iteration += 1        
    
    return policy, V, iteration

def main():
    n = 20
    k = 19
    m = 100
    random_seed = 2
    gamma = 0.9
    theta = 1e-6

    env = SimpleMazeGrid(n=n, k=k, m=m, render_option=True, random_seed=random_seed)
    env.render()

    policy, V, iteration = policy_iteration(env, gamma=gamma, theta=theta)

    # Plot the value function
    env.render_v_values(V, policy, iteration)
    plt.imshow(V.reshape((n, n)), cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Value Function')
    plt.savefig('policy_iteration_simple_maze_grid.png')
    plt.show()

    print("Completed!")

if __name__ == "__main__":
    main()