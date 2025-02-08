# Mahdi Rahbar

from source import FrozenLake
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk


# Create an environment
max_iter_number = 1000
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

env = FrozenLake(render_mode="human", map_name="8x8")
observation, info = env.reset(seed=30)



def create_reward_map(env):
    reward_map = np.zeros(env.nS)
    for s in range(env.nS):
        row, col = np.unravel_index(s, env.shape)

        if env._lake[row, col]:
            reward_map[s] = -10
        elif s == env.nS - 1:
            reward_map[s] = 10
        else:
            reward_map[s] = -1
    return reward_map



def value_iteration(env, gamma=0.85, threshold=1e-6):

    reward_map = create_reward_map(env)
    value_table = np.zeros(env.nS)

    slip_probs = {
        0: [(0.5, 0), (0.25, 1), (0.25, 3)],
        1: [(0.5, 1), (0.25, 0), (0.25, 2)],
        2: [(0.5, 2), (0.25, 1), (0.25, 3)],
        3: [(0.5, 3), (0.25, 0), (0.25, 2)]
    }


    iteration_count = 0

    for _ in range(max_iter_number):
        updated_value_table = np.copy(value_table)
        iteration_count += 1

        for s in range(env.nS):

            Q_values = []
            for a in range(env.nA):
                q_value = 0
                for slip_prob, slip_action in slip_probs[a]:
                    for prob, s_, _, _ in env.P[s][slip_action]:
                        if isinstance(s_, tuple):
                            s_ = 8 * s_[0] + s_[1]
                        q_value += slip_prob * (reward_map[s] + gamma * updated_value_table[s_])
                Q_values.append(q_value)
            value_table[s] = max(Q_values)

        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            break

    print(f"Value Iteration converged in {iteration_count} iterations.")
    return value_table



def extract_policy(env, value_table, gamma=0.85):
    policy = np.zeros(env.nS, dtype=int)

    reward_map = create_reward_map(env)

    slip_probs = {
        0: [(0.5, 0), (0.25, 1), (0.25, 3)],
        1: [(0.5, 1), (0.25, 0), (0.25, 2)],
        2: [(0.5, 2), (0.25, 1), (0.25, 3)],
        3: [(0.5, 3), (0.25, 0), (0.25, 2)]
    }

    for s in range(env.nS):
        Q_values = []
        for a in range(env.nA):
            q_value = 0
            for slip_prob, slip_action in slip_probs[a]:
                for prob, s_, _, _ in env.P[s][slip_action]:

                    if isinstance(s_, tuple):
                        s_ = 8 * s_[0] + s_[1]

                    q_value += slip_prob * (reward_map[s] + gamma * value_table[s_])
            Q_values.append(q_value)
        policy[s] = np.argmax(Q_values)

    return policy



def plot_heatmap(value_table, env_shape):

    reshaped_table = value_table.reshape(env_shape)

    plt.figure(figsize=(8, 8))
    plt.imshow(reshaped_table, cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('Value Heatmap')

    for i in range(env_shape[0]):
        for j in range(env_shape[1]):
            plt.text(j, i, f'{reshaped_table[i, j]:.2f}',
                     ha='center', va='center', color='black', fontsize=10)

    plt.show()




def compute_value_function(policy, env, gamma=0.85, threshold=1e-6):

    value_table = np.zeros(env.nS)

    reward_map = create_reward_map(env)

    slip_probs = {
        0: [(0.5, 0), (0.25, 1), (0.25, 3)],
        1: [(0.5, 1), (0.25, 0), (0.25, 2)],
        2: [(0.5, 2), (0.25, 1), (0.25, 3)],
        3: [(0.5, 3), (0.25, 0), (0.25, 2)]
    }

    while True:
        updated_value_table = np.copy(value_table)

        for state in range(env.nS):

            action = policy[state]
            q_value = 0

            for slip_prob, slip_action in slip_probs[action]:

                for trans_prob, next_state, _, _ in env.P[state][slip_action]:

                    if isinstance(next_state, tuple):
                        next_state = 8 * next_state[0] + next_state[1]

                    q_value += slip_prob * (reward_map[state] + gamma * updated_value_table[next_state])

            value_table[state] = q_value

        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            break

    return value_table




def policy_iteration(env, gamma=0.85, threshold=1e-6):

    policy = np.zeros(env.nS, dtype=int)

    iteration_count = 0

    for i in range(max_iter_number):

        iteration_count += 1

        value_table = compute_value_function(policy, env, gamma, threshold)

        new_policy = extract_policy(env, value_table, gamma)

        if np.array_equal(policy, new_policy):
            print(f"Policy Iteration converged in {iteration_count} iterations.")
            break

        policy = new_policy

    return policy, value_table




def run_value_iteration():
    global observation
    optimal_value_table = value_iteration(env)
    optimal_policy = extract_policy(env, optimal_value_table)
    plot_heatmap(optimal_value_table, env.shape)

    while True:
        action = optimal_policy[observation]
        next_state, reward, done, truncated, info = env.step(action)

        observation = 8 * next_state[0] + next_state[1]

        if observation == 63:
            print("You reached the goal! Returning to the main menu.")
            env.close()
            main_menu()
            break
        elif env._lake[next_state]:
            print("You fell into a hole! Restarting the game.")
            observation, info = env.reset()



def run_policy_iteration():
    global observation

    optimal_policy, value_table = policy_iteration(env)

    plot_heatmap(value_table, env.shape)

    while True:
        action = optimal_policy[observation]
        next_state, reward, done, truncated, info = env.step(action)

        observation = 8 * next_state[0] + next_state[1]

        if observation == 63:
            print("You reached the goal! Returning to the main menu.")
            env.close()
            main_menu()
            break
        elif env._lake[next_state]:
            print("You fell into a hole! Restarting the game.")
            observation, info = env.reset()




def main_menu():
    global env, observation

    def start_value_iteration():
        root.destroy()
        global env, observation
        env = FrozenLake(render_mode="human", map_name="8x8")
        observation, info = env.reset(seed=30)
        run_value_iteration()

    def start_policy_iteration():
        root.destroy()
        global env, observation
        env = FrozenLake(render_mode="human", map_name="8x8")
        observation, info = env.reset(seed=30)
        run_policy_iteration()


    root = tk.Tk()
    root.title("Main Menu")
    root.geometry("400x200")


    tk.Label(root, text="Choose Algorithm", font=("Arial", 16)).pack(pady=20)
    tk.Button(root, text="Value Iteration", font=("Arial", 14), command=start_value_iteration).pack(pady=10)
    tk.Button(root, text="Policy Iteration", font=("Arial", 14), command=start_policy_iteration).pack(pady=10)

    root.mainloop()



main_menu()

