import gymnasium as gym  # type: ignore
import numpy as np  # type: ignore

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

num_episodes = 1000  
gamma = 0.99          # Discount factor
threshold = 1e-6      # Convergence threshold for value iteration

num_states = env.observation_space.n
num_actions = env.action_space.n
goal_state = 15  

transition_counts = np.zeros((num_states, num_actions, num_states))
reward_sums = np.zeros((num_states, num_actions, num_states))

def manhattan_distance(state):
    grid_size = 4  
    x, y = divmod(state, grid_size)
    goal_x, goal_y = divmod(goal_state, grid_size)
    return abs(x - goal_x) + abs(y - goal_y)

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        transition_counts[state, action, next_state] += 1
        reward_sums[state, action, next_state] += reward
        state = next_state

T = np.zeros((num_states, num_actions, num_states))
R = np.zeros((num_states, num_actions, num_states))

for s in range(num_states):
    for a in range(num_actions):
        total_transitions = np.sum(transition_counts[s, a])
        if total_transitions > 0:
            T[s, a] = transition_counts[s, a] / total_transitions
            R[s, a] = np.divide(reward_sums[s, a], transition_counts[s, a], out=np.zeros_like(reward_sums[s, a]), where=transition_counts[s, a] > 0)

V = np.zeros(num_states)

while True:
    delta = 0
    new_V = np.copy(V)
    for s in range(num_states):
        max_value = float('-inf')
        for a in range(num_actions):
            value = np.sum(T[s, a] * (R[s, a] + gamma * V))
            max_value = max(max_value, value)
        new_V[s] = max_value
        delta = max(delta, abs(new_V[s] - V[s]))
    V = new_V
    if delta < threshold:
        break

policy = np.zeros(num_states, dtype=int)

for s in range(num_states):
    best_action = None
    best_value = float('-inf')
    for a in range(num_actions):
        next_states = np.where(T[s, a] > 0)[0]  
        for next_state in next_states:
            distance_reduction = manhattan_distance(s) - manhattan_distance(next_state)
            value = np.sum(T[s, a] * (R[s, a] + gamma * V))
            if (distance_reduction > 0 and value >= best_value) or (best_action is None):
                best_value = value
                best_action = a
    policy[s] = best_action if best_action is not None else np.argmax(
        [np.sum(T[s, a] * (R[s, a] + gamma * V)) for a in range(num_actions)]
    )

print("\nOptimal Policy (4x4 Grid):")
policy_grid = policy.reshape(4, 4)
for row in policy_grid:
    print(' '.join(map(str, row)))

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
state, _ = env.reset()
done = False

while not done:
    action = policy[state]
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()