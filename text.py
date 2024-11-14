import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import time
# get a queue of the events
from collections import deque

class ScreenSaverEnv(gym.Env):
    def __init__(self, canvas_size=(800, 600), frame_size=(100,100),image_path="path_to_image.png", speed=5):
        super(ScreenSaverEnv, self).__init__()
        
        # Initialize canvas and image properties
        self.canvas_width, self.canvas_height = canvas_size
        self.image = pygame.image.load(image_path)
        self.speed = speed
        # Resize image to fit within the canvas
        resize = (100, 100)
        # Apply the resize to the image
        self.image = pygame.transform.scale(self.image, resize)
        self.image_width, self.image_height = self.image.get_size()
        
        # Initialize starting position and velocity
        self.position = np.array([np.random.randint(0, self.canvas_width - self.image_width), 
                                  np.random.randint(0, self.canvas_height - self.image_height)])
        self.velocity = np.array([speed/2, speed/2])
        
        # Action space: No action needed, movement is automated
        self.action_space = spaces.Discrete(5)  # Dummy action space
        # Observation space: RGB array of the canvas
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        # Coords of the frame that captures the image
        self.frame_coords = np.array([canvas_size[0]//2, canvas_size[1]//2])
        self.frame_size = np.array(frame_size)
        self.frame_vel = np.array([0, 0])
        self.reward = 0
        self.counter = 0
        
        # Queue of the events
        self.events = deque(maxlen=100)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(canvas_size)
        pygame.display.set_caption("Screensaver Environment")

    def reset(self):
        # Initialize starting position and velocity
        self.position = np.array([np.random.randint(0, self.canvas_width - self.image_width), 
                                  np.random.randint(0, self.canvas_height - self.image_height)])
        self.velocity = np.array([self.speed/2, self.speed/2])
        self.counter = 0

        self.frame_coords = np.array([self.canvas_width//2, self.canvas_height//2])
        self.frame_vel = np.array([0, 0])
        self.reward = 0
        self.counter = 0

        
        return self._get_observation(), {}

    def step(self, action):
        # Update position
        self.apply_velocity(action)
        # Calculate reward
        reward = self.get_reward()
        self.reward = reward
        # Add the reward to the queue
        self.events.append(reward)
        
        # Bounce off the walls
        if self.position[0] <= 0 or self.position[0] + self.image_width >= self.canvas_width:
            self.velocity[0] = -self.velocity[0]
        if self.position[1] <= 0 or self.position[1] + self.image_height >= self.canvas_height:
            self.velocity[1] = -self.velocity[1]
        
        # No specific reward; observation only
        # [Observation, Reward, Done, Trunc, Info]
        self.counter += 1
        #print(f"Step #:{self.counter}")
        return self._get_observation(), reward, self.done(), False, {}

    # Method that apply the velocity to the position with bounce
    def apply_velocity(self, action):
        # Choose the velocity
        spd = self.speed
        velocity = [[0, 0], [0, spd], [spd, 0], [-spd, 0], [0, -spd]]
        vel = velocity[action]
        # Check bounds
        if self.position[0] + vel[0] <= 0 or self.position[0] + vel[0] + self.image_width >= self.canvas_width:
            vel[0] = -vel[0]
        if self.position[1] + vel[1] <= 0 or self.position[1] + vel[1] + self.image_height >= self.canvas_height:
            vel[1] = -vel[1]
        if self.frame_coords[0] + self.frame_vel[0] <= 0 or self.frame_coords[0] + self.frame_vel[0] + self.frame_size[0] >= self.canvas_width:
            self.frame_vel[0] = -self.frame_vel[0]
        if self.frame_coords[1] + self.frame_vel[1] <= 0 or self.frame_coords[1] + self.frame_vel[1] + self.frame_size[1] >= self.canvas_height:
            self.frame_vel[1] = -self.frame_vel[1]

        # Update position
        np.add(self.position, self.velocity, out=self.position, casting="unsafe")
        # Apply the velocity to the frame
        self.frame_vel = np.array(vel) 
        np.add(self.frame_coords, self.frame_vel, out=self.frame_coords, casting="unsafe")
        
    def _get_observation(self):
        # [Observation Format]: [Screen, Position, Velocity, Lebron Position, Lebron Velocity]
        # Render the current frame as an observation
        self.screen.fill((0, 0, 0))  # Clear screen with black
        self.screen.blit(self.image, self.position)  # Draw image at current position
        # Now add a hollow rectangle around the image as the frame
        pygame.draw.rect(self.screen, (255, 255, 255), (*self.frame_coords, self.image_width, self.image_height), 1)
        pygame.display.flip()  # Update display
        # Display the porcentage of the image inside the frame
        font = pygame.font.Font(None, 36)
        text = font.render(f"Reward: {self.reward:.2f}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        # Get the positions and velocities
        # Get the position of the image
        position = self.position / np.array([self.canvas_width, self.canvas_height])
        # Get the velocity of the image
        velocity = self.velocity
        # Get the position of the frame
        frame_position = self.frame_coords / np.array([self.canvas_width, self.canvas_height])
        # Get the velocity of the frame
        frame_velocity = self.frame_vel
        # Concatenate the positions and velocities
        observation = np.concatenate([position, velocity, frame_position, frame_velocity])
        
        # Convert the pygame screen to a numpy array
        frame = pygame.surfarray.array3d(self.screen)
        return (np.transpose(frame, (1, 0, 2)),observation)  # Transpose to match Gym's (H, W, C) format
    
    # The reward function
    def get_reward(self):
        # Check the porcentage of the image inside the frame
        # Get the area of the image
        image_area = self.image_width * self.image_height
        # Get the area of the frame
        frame_area = self.frame_size[0] * self.frame_size[1]
        # Get the intersection area
        intersection_area = max(0, min(self.frame_coords[0] + self.frame_size[0], self.position[0] + self.image_width) - max(self.frame_coords[0], self.position[0])) * max(0, min(self.frame_coords[1] + self.frame_size[1], self.position[1] + self.image_height) - max(self.frame_coords[1], self.position[1]))
        # Get the union area
        union_area = image_area + frame_area - intersection_area
        # Get the percentage of the image inside the frame
        percentage = intersection_area / union_area
        if 
        # Return the percentage
        return percentage
    
    # The done function
    def done(self):
        if np.mean(self.events) > 0.5 or self.counter>=200:
            return True
        else:
            return False
        

    def render(self, mode="human"):
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._get_observation()
        else:
            super(ScreenSaverEnv, self).render(mode)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
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