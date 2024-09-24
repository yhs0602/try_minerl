import gymnasium as gym
from crafter import Env


class SB3Wrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        print(f"{env.observation_space.shape=} {env.observation_space.dtype=}")
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Discrete(env.action_space.n)
        if env.metadata is None:
            env.metadata = {}
        if "semantics.async" not in env.metadata:
            env.metadata["semantics.async"] = False
        env.metadata["render_fps"] = 5

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info
