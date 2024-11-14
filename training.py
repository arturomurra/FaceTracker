import torch
import torch.optim as optim
from model.net import FullModel
from text import ScreenSaverEnv
import numpy as np
from torchvision import transforms  # Import for transformation
from PIL import Image
from main import Rollout, compute_returns


# # Cargamos el ciclo de entrenamiento
def train(model, env, optimizer, gamma=0.95, resize_resolution=(64, 64), k_epochs=4):
    # Collect data from the environment (Rollout)
    images, states, actions,old_probs, rewards, masks, values, entropies = Rollout(model, env, resize_resolution=resize_resolution)
    old_probs = old_probs.detach()
    values = values.detach()
    rewards = rewards.detach()
    actions = actions.detach()
    entropies = entropies.detach()
    for _ in range(k_epochs):
        # Calculate the loss
        dist, values = model(images, states)  # Get the distribution (policy) and values (critic)
        values = values.squeeze()
        # Calculate returns and advantages
        returns = compute_returns(rewards, masks, gamma)
        returns = returns.detach()
        returns = (returns - returns.mean()) / returns.std()
        advantages = returns - values


        # Policy loss (actor)
        log_probs = dist.log_prob(actions)  # Calculate log probability of actions
        ratio = torch.exp(log_probs - old_probs)
        policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1-0.2, 1+0.2) * advantages)
        

        # Value loss (critic)
        value_loss = 0.5 * advantages.pow(2)  # Critic loss (mean squared error)

        # Entropy loss (encouraging exploration)
        entropy_loss = -0.001 * entropies  # Entropy loss (scaled to encourage exploration)

        # Total loss
        total_loss = policy_loss.mean() + value_loss.mean() +entropy_loss.mean()

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()  # Compute gradients
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()  # Update model parameters

    return total_loss.item(), rewards


# Testeamos el entrenamiento
epochs = 100  # Número de épocas
scores = []  # Puntajes
moving_scores = []  # Puntaciones moviles
lr = 0.0001  # Learning rate
model = FullModel()  # Initialize model (fix initialization)
optimizer = optim.Adam(model.parameters(), lr=lr) 
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)

# Define image size for resizing
resize_resolution = (64, 64)  # Resize the images to 64x64

for epoch in range(epochs):
    # Entrenamiento
    loss, rewards = train(model, env, optimizer, gamma=0.95, resize_resolution=resize_resolution)

    # Calcular la puntuacion
    score = rewards.sum()  # Calculamos la puntuacion
    scores.append(score.item())  # Guardamos la puntuacion
    moving_scores.append(np.mean(scores[-100:]))  # Guardamos la puntuacion movil

    if epoch % 2== 0:
        print(f"Epoch: {epoch}")
        print(f"Score: {score}")
        print(f"Moving score: {moving_scores[-1]}")

# Fin del entrenamiento
