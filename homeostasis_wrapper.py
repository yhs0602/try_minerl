from typing import SupportsFloat, Any, Optional, Tuple, Dict

import gymnasium
from crafter import Env
from gymnasium.core import WrapperActType, WrapperObsType


class HomeostasisWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env, baseline=(9, 9, 9, 9), learning_rate=0.01):
        super().__init__(env)
        self.baseline = baseline
        self.weights = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        self.previous_homeostasis = 0
        self.learning_rate = learning_rate

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, external_reward, terminated, truncated, info = self.env.step(action)

        current_state = self.extract_agent_state(info)

        # Use L2 norm for drive calculation
        drive = [
            (current_state[i] - self.baseline[i]) ** 2
            for i in range(len(self.baseline))
        ]

        current_homeostasis = -sum(
            [self.weights[i] * drive[i] for i in range(len(drive))]
        )

        total_reward = float(external_reward) + (
            current_homeostasis - self.previous_homeostasis
        )  # Make type checker happy

        # print(f"{current_state=} {drive=} {current_homeostasis=} {self.previous_homeostasis=} {total_reward=} {self.weights=}")

        # Update weights based on homeostasis change (gradient descent step)
        for i in range(len(self.weights)):
            # Here, we assume the agent seeks to minimize homeostasis change
            gradient = -drive[i] * (current_homeostasis - self.previous_homeostasis)
            self.weights[i] -= self.learning_rate * gradient

        # Normalize weights to sum to 1
        self.weights = [w / sum(self.weights) for w in self.weights]

        self.previous_homeostasis = current_homeostasis
        return obs, total_reward, terminated, truncated, info

    def reset(
        self,
        *args,
        **kwargs,
    ):
        self.previous_homeostasis = 0
        return self.env.reset(*args, **kwargs)

    def extract_agent_state(self, info: Dict[str, Any]) -> Tuple[int, int, int, int]:
        inventory = info["inventory"]
        return (
            inventory["health"],
            inventory["food"],
            inventory["drink"],
            inventory["energy"],
        )
