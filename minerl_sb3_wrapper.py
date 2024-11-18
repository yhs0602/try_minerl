import gym
import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType
from typing import SupportsFloat, Any, Tuple, Dict


class TreeWrapper(gymnasium.Wrapper):
    def __init__(self, env: gym.Env):
        self.env = env
        super().__init__(env)
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [
                2,  # forward
                2,  # back
                2,  # left
                2,  # right
                2,  # jump
                2,  # sneak
                2,  # sprint
                2,  # attack
                25,  # pitch
                25,  # yaw
            ]
        )

    def step(
        self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        action_v2 = self.env.action_space.sample()
        action_v2["forward"] = action[0]
        action_v2["back"] = action[1]
        action_v2["left"] = action[2]
        action_v2["right"] = action[3]
        action_v2["jump"] = action[4]
        action_v2["sneak"] = action[5]
        action_v2["sprint"] = action[6]
        action_v2["attack"] = action[7]
        action_v2["camera_pitch"] = (action[8] - 12) * 15
        action_v2["camera_yaw"] = (action[9] - 12) * 15
        return self.env.step(action_v2)
