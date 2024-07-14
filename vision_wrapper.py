import gymnasium as gym
import numpy as np


class VisionWrapper(gym.Wrapper):
    def __init__(self, env, x_dim, y_dim):
        super().__init__(env)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(y_dim, x_dim, 3),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        obs = self.env.reset(**kwargs)
        return obs["pov"], {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs["pov"], info
