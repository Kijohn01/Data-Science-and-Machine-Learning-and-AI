import gymnasium as gym
import ale_py
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import cv2
import matplotlib.pyplot as plt


# Preprocess the observation (convert to grayscale and resize)


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (84, 84))  # Resize to 84x84
    return resized / 255.0  # Normalize the pixel values

# Build the Deep Q-Network (DQN)


def build_q_network(input_shape, n_actions):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(n_actions, activation='linear')
    ])
    return model

# Epsilon-greedy action selection


def epsilon_greedy_policy(model, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)  # Random action for exploration
    # Predict Q-values for the current state
    q_values = model.predict(state[None])
    print(np.argmax(q_values[0]))
    return np.argmax(q_values[0])  # Select the action with the highest Q-value


# Replay Buffer to store experiences


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def store(self, experience):
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove the oldest experience
        self.buffer.append(experience)  # Store the new experience

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Training the DQN


def train_dqn(env, n_episodes=700, batch_size=32, epsilon_start=1.0, epsilon_min=0.1, epsilon_decay=0.90, gamma=0.99, learning_rate=0.0025):
    n_actions = env.action_space.n  # Number of possible actions
    # Image size after preprocessing (84x84 grayscale)
    input_shape = (84, 84, 1)
    model = build_q_network(input_shape, n_actions)  # Build the Q-network

    # Target model for stable learning
    target_model = build_q_network(input_shape, n_actions)
    # Initialize target model with the same weights
    target_model.set_weights(model.get_weights())

    # Optimizer for the Q-network
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = ReplayBuffer()  # Initialize the replay buffer

    epsilon = epsilon_start  # Exploration factor
    total_rewards = []  # To store rewards for each episode

    for episode in range(n_episodes):
        state, _ = env.reset(seed=42)  # Reset the environment
        state = preprocess_frame(state)  # Preprocess the observation
        state = np.expand_dims(state, axis=-1)  # Add channel dimension
        done = False
        episode_reward = 0
        actions_taken = 0

        while not done:
            # update the action every 5 frames. Should improve performance
            if actions_taken % 3 == 0:
                # Select action using epsilon-greedy policy
                action = epsilon_greedy_policy(
                    model, state, epsilon, n_actions)

            actions_taken += 1

            next_state, reward, terminated, truncated, info = env.step(
                action)  # Step through the environment

            # exit loop condition
            done = terminated
            # Preprocess the next state

            next_state = preprocess_frame(next_state)
            next_state = np.expand_dims(
                next_state, axis=-1)  # Add channel dimension
            # Store experience in replay buffer
            replay_buffer.store(
                (state, action, reward, next_state, terminated or truncated))

            state = next_state  # Update state
            episode_reward += reward  # Update reward for the episode

            if replay_buffer.size() > batch_size:
                # Sample a batch of experiences
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Convert to numpy arrays
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)

                # Predict Q-values for the states
                q_values = model.predict(states)
                # Predict Q-values for the next states
                q_values_next = target_model.predict(next_states)

                # Update Q-values using the Bellman equation
                for i in range(batch_size):
                    if dones[i]:
                        # No future reward if the episode ended
                        q_values[i, actions[i]] = rewards[i]
                    else:
                        q_values[i, actions[i]] = rewards[i] + gamma * \
                            np.max(q_values_next[i])  # Bellman update

                # Apply the gradient update
                with tf.GradientTape() as tape:
                    # Predict Q-values for the states
                    q_values_pred = model(states)
                    loss = tf.reduce_mean(
                        tf.square(q_values_pred - q_values))  # Compute the loss
                # Compute gradients
                grads = tape.gradient(loss, model.trainable_variables)
                # Apply gradients to update model
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

        # Update the target model every 5 episodes
        if episode % 5 == 0:
            target_model.set_weights(model.get_weights())
            model.save_weights('FroggerModelWeights.h5')

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Store the total reward for the episode
        total_rewards.append(episode_reward)
        print(
            f"Episode {episode}/{n_episodes}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

    return model, total_rewards

# Main function to run the training


def main():
    env = gym.make("ALE/Frogger-v5", obs_type="rgb",
                   render_mode="human")  # Initialize the environment

    mode = input("Please enter mode [Train/Test]: ")
    if mode == "Train":
        trained_model, rewards = train_dqn(env)  # Train the model

        # Plot the training rewards
        plt.plot(rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Training Rewards Over Time')
        plt.show()
    elif mode == "Test":
        input_shape = (84, 84, 1)
        model = build_q_network(input_shape, 5)
        model.load_weights("FroggerModelWeights.h5")
        observation, _ = env.reset(seed=42)

        for i in range(10000):
            observation1 = preprocess_frame(observation)
            state = np.expand_dims(observation1, axis=-1)

            action = epsilon_greedy_policy(
                model, state, 0, 5)
            if i % 3 == 0:
                action = 0
            print(action)
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

    env.close()  # Close the environment once done


# Run the main function
if __name__ == '__main__':
    main()
