from text import ScreenSaverEnv
from model.net import FullModel
import numpy as np
import time

# Example usage
print("Empezando")
env = ScreenSaverEnv(canvas_size=(800, 600), image_path="lebronpng.png", speed=5)
print("Cargando el modelo")
model = FullModel()
obs, info = env.reset()
done = False
while not done:
    try:
        # transposing the observation to match the model input
        image = obs[0]
        image = np.transpose(image, (2, 0, 1))
        state = obs[1]
        print ("obs:",image.shape)
        print("state:",state.shape)
        action = model(image, state)
        
        print("action:",action.shape)
        obs, reward, done, truncated, info = env.step(action.argmax().item())
        env.render()
        time.sleep(0.01)
    except KeyboardInterrupt:
        env.close()

env.close()