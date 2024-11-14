import torch
import torch.optim as optim
from model.net import FullModel
from text import ScreenSaverEnv
import numpy as np
from torchvision import transforms  # Import for transformation
from PIL import Image

# # Cargamos el entorno
print("Empezando")
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)
print("Cargando el modelo")
# # Creamos la red neuronal
model = FullModel()

# # Cargamos el optimizador
lr = 0.0001  # Learning rate
gamma = 0.95  # Factor de descuento
optimizer = optim.RMSprop(model.parameters(), lr=lr)  # Optimizer [RMSprop]

# # Recolección de datos
def Rollout(model, env, resize_resolution=(64, 64)):
    images = []  # List to store images
    states = []  # States (position, velocity)
    actions = []  # Actions taken by the agent
    rewards = []  # Rewards received
    masks = []  # Masks for episode termination
    values = []  # Value predictions from the critic
    entropies = []  # Entropies of the action distributions
    state, _ = env.reset()  # Reset environment

    # Define the transform to resize the image
    transform = transforms.Compose([
        transforms.ToPILImage(),          # Convert tensor to PIL Image
        transforms.Resize(resize_resolution) # Resize the image
    ])

    while True:
        image = state[0]  # Image observation from the environment
        state_values = state[1]  # Position and velocity values
        
        # Apply resizing to the image before passing it to the model
        image_resized = transform(image)  # Resize the image to smaller resolution
        image_tensor = transforms.ToTensor()(image_resized)  # Convert resized image back to tensor

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
        rewards.append(reward)
        masks.append(mask)
        values.append(value)
        entropies.append(entropy)
        
        state = next_state  # Update state
        if done or truncated:
            break  # End the episode if done or truncated

    # Convert lists to tensors
    images = torch.stack(images)  # Stack images into a tensor
    states = torch.cat(states)  # Concatenate state tensors
    actions = torch.cat(actions)  # Concatenate actions tensors
    rewards = torch.FloatTensor(rewards)  # Convert rewards to tensor
    masks = torch.FloatTensor(masks)  # Convert masks to tensor
    values = torch.cat(values)  # Concatenate value tensors
    entropies = torch.cat(entropies)  # Concatenate entropy tensors
    return images, states, actions, rewards, masks, values, entropies


# # Testeamos el Rollout
images, states, actions, rewards, masks, values, entropies = Rollout(model, env, resize_resolution=(64, 64))

print("Images:", images.shape)
print("States:", states.shape)
print("Actions:", actions.shape)
print("Rewards:", rewards.shape)
print("Masks:", masks.shape)
print("Values:", values.shape)
print("Entropies:", entropies.shape)


# # Cargamos el ciclo de entrenamiento
def train(model, env, optimizer, gamma=0.95):
    # Collect data from the environment (Rollout)
    images, states, actions, rewards, masks, values, entropies = Rollout(model, env)

    # Calculate the loss
    dist, values = model(images, states)  # Get the distribution (policy) and values (critic)
    
    # Calculate returns and advantages
    returns = compute_returns(rewards, masks, gamma)
    advantages = returns - values

    # Policy loss (actor)
    log_probs = dist.log_prob(actions)  # Calculate log probability of actions
    policy_loss = -log_probs * advantages.detach()  # Actor loss (negative to minimize)

    # Value loss (critic)
    value_loss = 0.5 * (returns - values).pow(2)  # Critic loss (mean squared error)

    # Entropy loss (encouraging exploration)
    entropy_loss = -0.01 * entropies  # Entropy loss (scaled to encourage exploration)

    # Total loss
    total_loss = policy_loss.mean() + value_loss.mean() + entropy_loss.mean()

    # Backpropagation and optimization
    optimizer.zero_grad()
    total_loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    return total_loss.item()


# # Computamos los retornos
def compute_returns(rewards, masks, gamma):
    returns = torch.zeros_like(rewards)  # Initialize tensor for returns
    future_return = 0  # Start with 0 future return
    for t in reversed(range(len(rewards))):  # Iterate over rewards in reverse
        future_return = rewards[t] + gamma * future_return * masks[t]  # Calculate future return with discount
        returns[t] = future_return  # Assign the computed return for this step
    return returns