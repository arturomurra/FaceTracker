from text import ScreenSaverEnv
from model.net import FullModel
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
        action = model(obs[0], obs[1])
        print(action.shape)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.01)
    except KeyboardInterrupt:
        env.close()

env.close()