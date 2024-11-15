import torch
import torch.optim as optim
from model.net import FullModel
from text import ScreenSaverEnv
import numpy as np
from torchvision import transforms
from PIL import Image
from main import Rollout, compute_returns
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import csv  # Import csv for saving rewards to file

# Training loop with batch processing
def train(model, env, optimizer, gamma=0.95, resize_resolution=(64, 64), k_epochs=4):
    # Collect data from the environment (Rollout)
    images, states, actions, old_probs, rewards, masks, values, entropies = Rollout(model, env, resize_resolution=resize_resolution)
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
        total_loss = policy_loss.mean() + value_loss.mean() + entropy_loss.mean()

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()  # Compute gradients
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()  # Update model parameters

        # Save rewards to CSV
        csv_file = 'rewards.csv'
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Total Rewards'])  # Header row
            for epoch, reward in enumerate(all_rewards):
                writer.writerow([epoch, reward])  # Save epoch and corresponding reward

        print(f"Rewards saved to {csv_file}")

    return total_loss.item(), rewards

# Test the training with batch processing
epochs = 100  # Number of epochs
scores = []  # Scores
moving_scores = []  # Moving scores
all_rewards = []  # List to store rewards for each epoch
lr = 0.0001  # Learning rate
model = FullModel()  # Initialize model
optimizer = optim.Adam(model.parameters(), lr=lr) 
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=3)

# Define image size for resizing
resize_resolution = (64, 64)  # Resize the images to 64x64

for epoch in range(epochs):
    # Training
    loss, rewards = train(model, env, optimizer, gamma=0.95, resize_resolution=resize_resolution)

    # Calculate score
    score = rewards.sum()  # Calculate the score
    scores.append(score.item())  # Store the score
    moving_scores.append(np.mean(scores[-100:]))  # Store the moving average score
    all_rewards.append(score.item())  # Append the total reward of the epoch

    if epoch % 2 == 0:
        print(f"Epoch: {epoch}")
        print(f"Score: {score}")
        print(f"Moving score: {moving_scores[-1]}")

# Plot the rewards using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, label='Total Rewards per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Total Rewards')
plt.title('Rewards Over Time')
plt.legend()

# Save the plot as a PNG file
plot_filename = 'rewards_plot.png'  # You can change the file name and extension as needed
plt.savefig(plot_filename)

# Show the plot
plt.show()

print(f"Plot saved as {plot_filename}")

# End of training
