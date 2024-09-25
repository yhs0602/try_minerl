import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TileFeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=7),  # 타일 단위로 CNN을 적용
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class FeatureExtractorV2(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
    ):
        super(FeatureExtractorV2, self).__init__(
            observation_space=observation_space, features_dim=512
        )
        # Given 7x7x3 input pixels, output a 64-dimensional feature vector
        # Input: (7x9, 7, 7, 3); (batch, height, width, channels)
        # Output: (7x9, 64); (batch, features)
        self.ingame_base_feature_cnn = TileFeatureExtractor()
        self.ingame_feature_extractor = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # 타일 피처맵을 다시 CNN으로 처리
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPooling으로 공간 차원을 줄임
        )
        # Given 7x7x3 input pixels, output a 64-dimensional feature vector
        # Input: (2x9, 7, 7, 3); (batch, height, width, channels)
        # Output: (2x9, 64); (batch, features)
        self.hud_feature_extractor = nn.Sequential(
            TileFeatureExtractor(),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # 타일 피처맵을 다시 CNN으로 처리
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPooling으로 공간 차원을 줄임
        )

        # 7x9 tiles -> a single feature (dim: 256)
        self.player_orientation_extractor = nn.Sequential(
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.Softmax(dim=-1),  # 없어도 될 수도 있음.
        )
        self.linear = nn.Linear(256 * 3 * 4 + 256 * 1 * 4 + 4, 512)

        # 256 + 256 + 4 -> 516
        # Actor와 Critic
        # self.actor = nn.Sequential(
        #     nn.Linear(516, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 4),
        # )
        #
        # self.critic = nn.Sequential(
        #     nn.Linear(516, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        # )

    def forward(self, image):
        # image.shape=(batch, channels, height, width)
        ingame_tiles = image[:, :, :49, :63]
        hud_tiles = image[:, :, 49:63, :63]
        # print(f"{ingame_tiles.shape=} {hud_tiles.shape=} {image.shape=}")
        ingame_base_features = self.ingame_base_feature_cnn(ingame_tiles)
        ingame_features = self.ingame_feature_extractor(ingame_base_features)
        # print(f"{ingame_base_features.shape=} {ingame_features.shape=}")
        center_feature = self.player_orientation_extractor(
            ingame_base_features[:, :, 3, 4]
        )
        hud_features = self.hud_feature_extractor(hud_tiles)
        # print(f"{ingame_features.shape=} {hud_features.shape=}")

        # Concatenate global features
        # print(f"{global_hud_feature.shape=} {global_ingame_feature.shape=} {center_feature.shape=}")
        global_features = torch.cat(
            [
                ingame_features.reshape(-1, 256 * 3 * 4),
                center_feature,
                hud_features.reshape(-1, 256 * 1 * 4),
            ],
            dim=-1,
        )
        # print(f"{global_features.shape=}")
        return self.linear(global_features)
        # # Use global features to predict action
        # action_prob = self.actor(global_features)
        # # Use global features to predict value
        # value = self.critic(global_features)
        # return action_prob, value
