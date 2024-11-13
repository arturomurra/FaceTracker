import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time

class ScreenSaverEnv(gym.Env):
    def __init__(self, canvas_size=(800, 600), frame_size=(100,100),image_path="path_to_image.png", speed=5):
        super(ScreenSaverEnv, self).__init__()
        
        # Initialize canvas and image properties
        self.canvas_width, self.canvas_height = canvas_size
        self.image = pygame.image.load(image_path)
        # Resize image to fit within the canvas
        resize = (100, 100)
        # Apply the resize to the image
        self.image = pygame.transform.scale(self.image, resize)
        self.image_width, self.image_height = self.image.get_size()
        
        # Initialize starting position and velocity
        self.position = np.array([np.random.randint(0, self.canvas_width - self.image_width), 
                                  np.random.randint(0, self.canvas_height - self.image_height)])
        self.velocity = np.array([speed, speed])
        
        # Action space: No action needed, movement is automated
        self.action_space = spaces.Discrete(5)  # Dummy action space
        # Observation space: RGB array of the canvas
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        # Coords of the frame that captures the image
        self.frame_coords = np.array([canvas_size[0]//2, canvas_size[1]//2])
        self.frame_size = np.array(frame_size)
        self
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(canvas_size)
        pygame.display.set_caption("Screensaver Environment")

    def reset(self):
        # Reset position to a random location
        self.position = np.array([np.random.randint(0, self.canvas_width - self.image_width), 
                                  np.random.randint(0, self.canvas_height - self.image_height)])
        # Reset velocity
        self.velocity = np.array([np.random.choice([-1, 1]), np.random.choice([-1, 1])]) * np.linalg.norm(self.velocity)
        
        return self._get_observation(), {}

    def step(self, action):
        # Update position
        self.apply_velocity()
        self.apply_frame(action)
        
        # Bounce off the walls
        if self.position[0] <= 0 or self.position[0] + self.image_width >= self.canvas_width:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] <= 0 or self.position[1] + self.image_height >= self.canvas_height:
            self.velocity[1] = -self.velocity[1]
        
        # No specific reward; observation only
        return self._get_observation(), 0, False, False, {}

    # Method that apply the velocity to the position with bounce
    def apply_velocity(self):
        # Update position
        np.add(self.position, self.velocity, out=self.position, casting="unsafe")

    # Method that apply the frame to the position
    def apply_frame(self, action):
        # Move the frame coords the action
        acelerations = [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
        # Action 1
        if action == 0:

            np.add(self.frame_coords, velocities[action], out=self.frame_coords, casting="unsafe")
        # Action 2
        if action == 1:
            np.add(self.frame_coords, velocities[action], out=self.frame_coords, casting="unsafe")
        # Action 3
        if action == 2:
            np.add(self.frame_coords, velocities[action], out=self.frame_coords, casting="unsafe")
        # Action 4
        if action == 3:
            np.add(self.frame_coords, velocities[action], out=self.frame_coords, casting="unsafe")
        # Action 5
        if action == 4:
            np.add(self.frame_coords, velocities[action], out=self.frame_coords, casting="unsafe")
        
        # Apply the frame to the position
        self.position = np.add(self.position, self.frame_coords, casting="unsafe")

        
    def _get_observation(self):
        # Render the current frame as an observation
        self.screen.fill((0, 0, 0))  # Clear screen with black
        self.screen.blit(self.image, self.position)  # Draw image at current position
        # Now add a hollow rectangle around the image as the frame
        pygame.draw.rect(self.screen, (255, 255, 255), (*self.frame_coords, self.image_width, self.image_height), 1)
        pygame.display.flip()  # Update display
        
        # Convert the pygame screen to a numpy array
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2))  # Transpose to match Gym's (H, W, C) format

    def render(self, mode="human"):
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._get_observation()
        else:
            super(ScreenSaverEnv, self).render(mode)

    def close(self):
        pygame.quit()

# Example usage
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)
obs, info = env.reset()
done = False
while not done:
    try:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)
    except KeyboardInterrupt:
        env.close()

env.close()