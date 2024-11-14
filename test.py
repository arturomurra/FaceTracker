import torch
import torch.optim as optim
import numpy as np
import time
from text import ScreenSaverEnv
from model.net import FullModel

# PPO parameters
gamma = 0.99  # Discount factor
epsilon = 0.2  # PPO clip parameter
lr = 0.001  # Learning rate
batch_size = 64  # Number of samples per batch

# Initialize environment and model
print("Empezando")
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)
print("Cargando el modelo")
model = FullModel()

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

# Function to calculate the PPO loss
def ppo_loss(old_log_probs, log_probs, returns, advantages):
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    return -torch.min(surr1, surr2).mean()

# Function to compute returns
def compute_returns(rewards, values, next_value, gamma):
    returns = []
    R = next_value
    for reward, value in zip(rewards[::-1], values[::-1]):
        R = reward + gamma * R
        returns.insert(0, R)
    return returns

# Training loop
obs, info = env.reset()
done = False
counter = 0
trajectory = []  # Collect experiences
while not done:
    try:
        # Transpose the observation to match the model input
        image = obs[0]
        image = np.transpose(image, (2, 0, 1))  # [Height, Width, Channel] -> [Channel, Height, Width]
        state = obs[1]

        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get action from the model
        action = model(image_tensor, state_tensor)
        action_probs = torch.softmax(action, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action.argmax(dim=-1))

        # Step the environment
        obs, reward, done, truncated, info = env.step(action.argmax().item())

        # Store transition
        trajectory.append((image_tensor, state_tensor, action, log_prob, reward))

        # Once enough experiences are collected, update the model
        if len(trajectory) >= batch_size:
            # Prepare batch data for training
            states, actions, log_probs, rewards = zip(*[(t[1], t[2], t[3], t[4]) for t in trajectory])
            states = torch.cat(states)
            actions = torch.cat(actions)
            log_probs = torch.cat(log_probs)
            rewards = torch.tensor(rewards, dtype=torch.float32)

            # Compute returns and advantages
            returns = compute_returns(rewards, states, 0, gamma)  # Assuming next value = 0 for terminal states
            returns = torch.tensor(returns)
            advantages = returns - values

            # Perform PPO update
            optimizer.zero_grad()
            loss = ppo_loss(log_probs, log_probs, returns, advantages)
            loss.backward()
            optimizer.step()

            print(f"Step Reward: {reward:.2f}, Loss: {loss.item():.4f}")

            # Clear trajectory after update
            trajectory = []

        # Render the environment
        env.render()

        time.sleep(0.01)
        counter += 1
    except KeyboardInterrupt:
        env.close()

env.close()
