from text import ScreenSaverEnv
from model.net import FullModel
import torch
import torch.optim as optim
import numpy as np
import time

# Example usage
print("Empezando")
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)
print("Cargando el modelo")
model = FullModel()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Loss function to minimize (Mean Squared Error for regression)

# Training loop
obs, info = env.reset()
done = False
counter = 0
while not done:
    try:
        # Transpose the observation to match the model input
        image = obs[0]
        image = np.transpose(image, (2, 0, 1))  # [Height, Width, Channel] -> [Channel, Height, Width]
        state = obs[1]
        
        # Convert to tensors
        image_tensor = torch.tensor(image, dtype=torch.float32)  # Add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Get action from the model (predicted accelerations)
        action, log = model.get_action(image_tensor, state_tensor)
        
        # Take action in the environment (based on model output)
        # In your case, we need to select the highest value from action (argmax)
        action_taken = action  # Convert to a scalar value representing the chosen action
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action_taken)
        
        # Calculate the loss: The target is the "correct" action, which you should define (e.g., based on image positioning)
        target = torch.tensor([reward], dtype=torch.float32)  # Reward as the target for training

        # Compute loss (MSE between predicted action and target)
        loss = criterion(action[0], target)  # Compare predicted action vs. actual action

        # Backpropagate the loss
        #optimizer.zero_grad()  # Zero the gradients
        #loss.backward()  # Compute gradients
        #optimizer.step()  # Update the model parameters

        # Print for monitoring
        print(f"Step Reward: {reward:.2f}, Loss: {loss.item():.4f}")
        
        # Render the environment
        env.render()
        
        
        counter = counter + 1

    except KeyboardInterrupt:
        env.close()

env.close()
