import gymnasium as gym
import numpy as np
import random

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", render_mode="human", map_name="4x4")

# Initialize Q-table with zeros
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
Q = np.zeros((state_space_size, action_space_size))

# Q-Learning parameters
alpha = 0.1       # Learning rate
gamma = 0.99      # Discount factor
epsilon = 1.0     # Initial exploration probability
epsilon_min = 0.01 # Minimum exploration probability
epsilon_decay = 0.995 # Decay rate for exploration probability
num_episodes = 1000  # Number of episodes for training

# Function to choose an action based on exploration
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore: choose a random action
    else:
        return np.argmax(Q[state])  # Exploit: choose the action with highest Q-value

# Training the Q-learning agent
for episode in range(num_episodes):
    state, info = env.reset(seed=42)  # Start a new episode
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(int(state))  # Choose an action
        next_state, reward, terminated, truncated, info = env.step(action)  # Take action
        next_state = int(next_state)  # Ensure next_state is an integer
        
        # Update Q-value using the Q-learning formula
        best_next_action = np.argmax(Q[next_state])
        Q[int(state), action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[int(state), action])
        
        state = next_state  # Move to the next state
        total_reward += reward
        
        done = terminated or truncated  # Check if the episode is finished

    # Update exploration probability
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Testing the trained policy
num_test_episodes = 10
for episode in range(num_test_episodes):
    state, info = env.reset(seed=42)  # Start a new episode
    done = False
    total_reward = 0
    print(f"Test Episode {episode + 1}")
    
    while not done:
        action = np.argmax(Q[int(state)])  # Choose the best action
        next_state, reward, terminated, truncated, info = env.step(action)  # Take action
        env.render()  # Show the environment
        
        next_state = int(next_state)  # Ensure next_state is an integer
        
        state = next_state  # Move to the next state
        total_reward += reward
        
        done = terminated or truncated  # Check if the episode is finished
    
    print(f"Test Episode finished with total reward: {total_reward}")

# Close the environment
env.close()
