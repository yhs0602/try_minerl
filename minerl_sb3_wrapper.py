import gym
import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType
from typing import SupportsFloat, Any, Tuple, Dict, Optional


class SB3MineRLWrapper(gymnasium.Wrapper):
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
        action_v2["camera"] = [(action[8] - 12) * 15, (action[9] - 12) * 15]
        action_v2["drop"] = 0
        action_v2["swapHands"] = 0
        action_v2["use"] = 0
        for i in range(1, 10):
            action_v2[f"hotbar.{i}"] = 0
        action_v2["ESC"] = 0
        return self.env.step(action_v2)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        obs = self.env.reset()
        if isinstance(obs, tuple):
            return obs
        else:
            # 결과가 튜플이 아니면 obs와 빈 딕셔너리를 반환
            return obs, {}
