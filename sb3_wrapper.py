import gymnasium as gym


class SB3Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        if env.metadata is None:
            env.metadata = {}
        if "semantics.async" not in env.metadata:
            env.metadata["semantics.async"] = False

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info
