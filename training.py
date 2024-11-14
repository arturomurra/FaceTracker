import torch
import torch.optim as optim
from model.net import FullModel
from text import ScreenSaverEnv
import numpy as np
from torchvision import transforms  # Import for transformation
from PIL import Image
from main import Rollout, compute_returns


# # Cargamos el ciclo de entrenamiento
def train(model, env, optimizer, gamma=0.95, resize_resolution=(64, 64)):
    # Collect data from the environment (Rollout)
    images, states, actions, rewards, masks, values, entropies = Rollout(model, env, resize_resolution=resize_resolution)

    # Calculate the loss
    dist, values = model(images, states)  # Get the distribution (policy) and values (critic)
    
    # Calculate returns and advantages
    returns = compute_returns(rewards, masks, gamma)
    returns = (returns - returns.mean())/(returns.std() + 1e-8)
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

    return total_loss.item(), rewards


# Testeamos el entrenamiento
epochs = 100  # Número de épocas
scores = []  # Puntajes
moving_scores = []  # Puntaciones moviles
lr = 0.0001  # Learning rate
model = FullModel()  # Initialize model (fix initialization)
optimizer = optim.RMSprop(model.parameters(), lr=lr) 
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)

# Define image size for resizing
resize_resolution = (64, 64)  # Resize the images to 64x64

for epoch in range(epochs):
    # Entrenamiento
    loss, rewards = train(model, env, optimizer, gamma=0.99, resize_resolution=resize_resolution)

    # Calcular la puntuacion
    score = rewards.sum()  # Calculamos la puntuacion
    scores.append(score.item())  # Guardamos la puntuacion
    moving_scores.append(np.mean(scores[-100:]))  # Guardamos la puntuacion movil

    if epoch % 2== 0:
        print(f"Epoch: {epoch}")
        print(f"Score: {score}")
        print(f"Moving score: {moving_scores[-1]}")

# Fin del entrenamiento
