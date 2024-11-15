import torch
import torch.optim as optim
from model.net import FullModel
from text import ScreenSaverEnv
import numpy as np
from torchvision import transforms  # Import for transformation
from PIL import Image


# # Recolección de datos
def Rollout(model, env, resize_resolution=(64, 64), nums=3):
    images = []  # List to store images
    states = []  # States (position, velocity)
    actions = []  # Actions taken by the agent
    logs_probs = []  # Log probabilities of the actions
    rewards = []  # Rewards received
    masks = []  # Masks for episode termination
    values = []  # Value predictions from the critic
    entropies = []  # Entropies of the action distributions


    # Define the transform to resize the image
    transform = transforms.Compose([
        transforms.ToPILImage(),          # Convert tensor to PIL Image
        transforms.Resize(resize_resolution) # Resize the image
    ])
    for num in range(nums):
        state, _ = env.reset()  # Reset environment
        while True:
            image = state[0]  # Image observation from the environment
            state_values = state[1]  # Position and velocity values
            
            # Apply resizing to the image before passing it to the model
            image_resized = transform(image)  # Resize the image to smaller resolution
            image_tensor = transforms.ToTensor()(image_resized).unsqueeze(0)  # Convert resized image back to tensor

            state_values_tensor = torch.FloatTensor(state_values).unsqueeze(0)  # Convert state values to tensor and add batch dimension

            dist, value = model(image_tensor, state_values_tensor)  # Get the distribution (for action selection) and value from the model
            action = dist.sample()  # Sample action based on the distribution
            entropy = dist.entropy()  # Get the entropy of the action distribution
            next_state, reward, done, truncated, _ = env.step(action.item())  # Take the action in the environment
            
            mask = 0 if done else 1  # If done, mask is 0, else 1 (for episode termination)
            
            # Store the data
            images.append(image_tensor)
            states.append(state_values_tensor)
            actions.append(action)
            logs_probs.append(dist.log_prob(action))  # Log probability of the action
            rewards.append(reward)
            masks.append(mask)
            values.append(value)
            entropies.append(entropy)
            
            state = next_state  # Update state
            if done or truncated:
                break  # End the episode if done or truncated

    # Convert lists to tensors
    images = torch.cat(images)  # Stack images into a tensor
    states = torch.cat(states)  # Concatenate state tensors
    actions = torch.cat(actions)  # Concatenate actions tensors
    logs_probs = torch.cat(logs_probs)  # Concatenate log probabilities
    rewards = torch.FloatTensor(rewards)  # Convert rewards to tensor
    masks = torch.FloatTensor(masks)  # Convert masks to tensor
    values = torch.cat(values)  # Concatenate value tensors
    entropies = torch.cat(entropies)  # Concatenate entropy tensors
    return images, states, actions, logs_probs, rewards, masks, values, entropies







# # Computamos los retornos
def compute_returns(rewards, masks, gamma):
    returns = torch.zeros_like(rewards)  # Initialize tensor for returns
    future_return = 0  # Start with 0 future return
    for t in reversed(range(len(rewards))):  # Iterate over rewards in reverse
        future_return = rewards[t] + gamma * future_return * masks[t]  # Calculate future return with discount
        returns[t] = future_return  # Assign the computed return for this step
    return returns