import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from text import ScreenSaverEnv
from model.net import FullModel

# Define PPO Model
class PPOModel(nn.Module):
    def __init__(self, action_space, observation_space):
        super(PPOModel, self).__init__()
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Policy Network (actor)
        self.policy_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(32 * (observation_space[0]//4) * (observation_space[1]//4), 256),
            nn.ReLU(),
            nn.Linear(256, action_space)
        )
        
        # Value Network (critic)
        self.value_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(32 * (observation_space[0]//4) * (observation_space[1]//4), 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        action_probs = self.policy_net(x)
        value = self.value_net(x)
        return action_probs, value

# Generalized Advantage Estimation (GAE) for calculating advantages
def compute_advantages(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    advantages = []
    returns = []
    next_value = 0  # Next value (done = 0 when at terminal state)
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantage = delta + gamma * lambda_ * (1 - dones[t]) * advantage if t < len(rewards) - 1 else delta
        advantages.insert(0, advantage)
        return_ = advantage + values[t]
        returns.insert(0, return_)
        next_value = values[t]
    
    return torch.tensor(advantages), torch.tensor(returns)

# PPO loss function with clipping, value loss, and entropy bonus
def ppo_loss(old_probs, new_probs, advantages, old_values, new_values, returns, epsilon=0.2, entropy_coeff=0.01):
    # Compute the ratio
    ratio = (new_probs / old_probs).clamp(1 - epsilon, 1 + epsilon)
    
    # Policy loss
    policy_loss = -torch.min(ratio * advantages, ratio * advantages.clamp(-epsilon, epsilon)).mean()

    # Value loss
    value_loss = F.mse_loss(new_values, returns)

    # Entropy loss (encourage exploration)
    entropy_loss = -(new_probs * torch.log(new_probs)).mean()

    return policy_loss + 0.5 * value_loss - entropy_coeff * entropy_loss

# Collecting experience from the environment
def collect_experience(env, model, num_steps=1000):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    state = env.reset()  # This returns a tuple (observation, info)
    
    for step in range(num_steps):
        # Extract the image and position/velocity from the state
        image = state[0]  # The image (canvas with the moving object)
        position_velocity = state[1]  # The position and velocity data
        
        # Ensure the image has the correct shape (C, H, W)
        image = np.transpose(image, (2, 0, 1))  # [H, W, C] -> [C, H, W]
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: [1, C, H, W]
        position_velocity_tensor = torch.tensor(position_velocity, dtype=torch.float32).unsqueeze(0)  # Shape: [1, N]
        
        # Get action probabilities and value predictions
        action_probs, value = model(image_tensor)
        
        # Sample an action from the action probabilities
        action_dist = torch.distributions.Categorical(F.softmax(action_probs, dim=-1))
        action = action_dist.sample()  # Sample action
        
        # Step the environment
        next_state, reward, done, _, _ = env.step(action.item())  # Take the action
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        
        if done:
            break

    return states, actions, rewards, next_states, dones


# Training loop for PPO
def train_ppo(env, model, optimizer, num_epochs=1000, batch_size=32):
    for epoch in range(num_epochs):
        states, actions, rewards, next_states, dones = collect_experience(env, model)
        
        # Convert collected data into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Calculate values and advantages
        old_action_probs, old_values = model(states)
        new_action_probs, new_values = model(states)
        
        # Calculate advantages and returns
        advantages, returns = compute_advantages(rewards, old_values, new_values, dones)
        
        # Calculate PPO loss
        old_probs = F.softmax(old_action_probs, dim=-1)
        new_probs = F.softmax(new_action_probs, dim=-1)
        
        loss = ppo_loss(old_probs, new_probs, advantages, old_values, new_values, returns)
        
        # Backpropagate and update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print for monitoring
        print(f"Epoch {epoch}/{num_epochs} - Loss: {loss.item()}")
        
        # Render the environment
        env.render()
        
        time.sleep(0.01)  # Slow down the loop a bit

# Example usage
if __name__ == "__main__":
    print("Empezando")
    
    # Initialize environment and model
    env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)
    model = PPOModel(action_space=5, observation_space=(600, 800, 3))
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # Train the PPO model
    train_ppo(env, model, optimizer)
    
    env.close()
