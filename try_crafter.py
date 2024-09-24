import crafter  # noqa
import gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from crafter_feature_extractor_v2 import FeatureExtractorV2
from get_device import get_device
from sb3_wrapper import SB3Wrapper

# class FeatureExtractor(BaseFeaturesExtractor):
# def __init__(
#             self, observation_space: gym.Space,
#     ):
#         super(FeatureExtractor, self).__init__(observation_space=observation_space, features_dim=516)
#         # Given 7x7x3 input pixels, output a 64-dimensional feature vector
#         # Input: (7x9, 7, 7, 3); (batch, height, width, channels)
#         # Output: (7x9, 64); (batch, features)
#         self.ingame_base_feature_cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
#         # Given 7x7x3 input pixels, output a 64-dimensional feature vector
#         # Input: (2x9, 7, 7, 3); (batch, height, width, channels)
#         # Output: (2x9, 64); (batch, features)
#         self.hud_base_feature_cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )
#
#         # 7x9 tiles -> a single feature (dim: 256)
#         self.global_ingame_feature_extractor = nn.Sequential(
#             nn.Conv2d(1600, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(8064, 256),
#             nn.ReLU(),
#         )
#
#         self.global_hud_feature_extractor = nn.Sequential(
#             nn.Conv2d(1600, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(2304, 256),
#             nn.ReLU(),
#         )
#
#         self.player_orientation_extractor = nn.Sequential(
#             nn.Linear(1600, 4),
#             nn.ReLU(),
#             nn.Softmax(dim=-1),  # 없어도 될 수도 있음.
#         )
#
#         # 256 + 256 + 4 -> 516
#         # Actor와 Critic
#         self.actor = nn.Sequential(
#             nn.Linear(516, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 4),
#         )
#
#         self.critic = nn.Sequential(
#             nn.Linear(516, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#         )
#
#     def forward(self, image):
#         # image.shape=(batch, height, width, channels)
#         # 64x64 image -> 9x9 tiles (7x7x3 pixels each)
#         # 9x9 tiles -> 2x9 tiles + 7x9 tiles
#         tiles = self.extract_tiles_batch(image, tile_size=7)
#         # print(f"{tiles.shape=}")
#         ingame_tiles = tiles[:, :, :7, :, :]  # 7x9 타일
#         hud_tiles = tiles[:, :, 7:, :, :]  # 2x9 타일
#         # print(f"{ingame_tiles.shape=}{hud_tiles.shape=}")
#         # center_tile = tiles[:, :, 3, 4, :]  # 중앙 타일
#         # print(f"{center_tile.shape=}")
#         ingame_tiles = ingame_tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 7, 7)
#         ingame_features = self.ingame_base_feature_cnn(ingame_tiles).reshape(-1, 7, 9, 1600).permute(0, 3, 1, 2)
#         hud_tiles = hud_tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 7, 7)
#         hud_features = self.hud_base_feature_cnn(hud_tiles).reshape(-1, 2, 9, 1600).permute(0, 3, 1, 2)
#         # Use ingame features to extract global ingame feature
#         global_ingame_feature = self.global_ingame_feature_extractor(ingame_features)
#         global_hud_feature = self.global_hud_feature_extractor(hud_features)
#         center_feature = self.player_orientation_extractor(ingame_features[:, :, 3, 4])
#         # Concatenate global features
#         # print(f"{global_hud_feature.shape=} {global_ingame_feature.shape=} {center_feature.shape=}")
#         global_features = torch.cat([global_ingame_feature, global_hud_feature, center_feature], dim=-1)
#         # print(f"{global_features.shape=}")
#         return global_features
#         # # Use global features to predict action
#         # action_prob = self.actor(global_features)
#         # # Use global features to predict value
#         # value = self.critic(global_features)
#         # return action_prob, value
#
#     # 7x7 타일을 추출하여 한 번에 배치로 처리하는 함수
#     def extract_tiles_batch(self, image: torch.Tensor, tile_size=7):
#         # image: (batch, channels, height, width)
#         # result: (batch, channels, num_tiles_y, num_tiles_x, tile_size, tile_size)
#         # print(f"Input image shape = {image.shape}")
#         unfolded1 = image.unfold(2, tile_size, tile_size)
#         # print(f"Unfolded1 shape = {unfolded1.shape}")
#         unfolded2 = unfolded1.unfold(3, tile_size, tile_size)
#         # print(f"Unfolded2 shape = {unfolded2.shape}")
#         # tiles.shape=torch.Size([1, 3, 9, 9, 7, 7])
#         # batch_size, color_channel, grid_x, grid_y, pixel_x, pixel_y?
#         return unfolded2
#         # return image.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)


if __name__ == "__main__":
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="crafter-test",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="v1",
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env = gym.make("CrafterReward-v1")  # Or CrafterNoReward-v1
    device = get_device(0)
    # env = crafter.Recorder(
    #     env, './path/to/logdir',
    #     save_stats=True,
    #     save_video=True,
    #     save_episode=False,
    # )
    env.render_mode = "rgb_array"
    env = Monitor(SB3Wrapper(env))
    env = DummyVecEnv([lambda: env])
    eval_callback = EvalCallback(
        VecVideoRecorder(
            env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % 10000 == 0,
            video_length=10000,
        ),
        best_model_save_path=f"models/{run.id}",
        log_path=f"logs/{run.id}",
        eval_freq=500,
        n_eval_episodes=30,
        deterministic=False,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs={
            "features_extractor_class": FeatureExtractorV2,
        },
    )
    try:
        model.learn(
            total_timesteps=1000000,
            callback=[
                # CustomWandbCallback(),
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                eval_callback,
            ],
        )
        model.save(f"ckpts/v1-{run.name}.ckpt")
    finally:
        env.close()
        run.finish()

    # obs = env.reset()
    # start_time = time.time_ns()
    # done = False
    # for i in range(9000000):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step([action])
    #     # obs.shape=(1, 64, 64, 3)
    #     time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
    #     fps = int(i / time_elapsed)
    #     if i % 64 == 0:
    #         wandb.log(
    #             {
    #                 "time/iterations": i,
    #                 "time/fps": fps,
    #                 "time/time_elapsed": int(time_elapsed),
    #                 "time/total_timesteps": i,
    #             }
    #         )
    #     if i % 4000 == 0:
    #         print(f"Step: {i}")
    #     if done:
    #         break
    # env.close()
    # run.finish()
