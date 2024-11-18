import argparse
import logging

import coloredlogs
import minerl.env._singleagent as _singleagent
import minerl.herobraine.env_specs.basalt_specs as basalt_specs
import minerl.herobraine.env_specs.obtain_specs as minerl_herobraine_envs
import wandb
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from get_device import get_device
from minerl_sb3_wrapper import SB3MineRLWrapper
from vision_wrapper import VisionWrapper

coloredlogs.install(logging.DEBUG)


# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)
class ResizableObtainDiamondShovelEnvSpec(
    minerl_herobraine_envs.ObtainDiamondShovelEnvSpec
):
    def __init__(self, resolution=[640, 360]):
        minerl_herobraine_envs.HumanSurvival.__init__(
            self,
            name=f"MineRLObtainDiamondShovel-v0-res-{resolution[0]}-{resolution[1]}",
            max_episode_steps=minerl_herobraine_envs.TIMEOUT,
            # Hardcoded variables to match the pretrained models
            fov_range=[70, 70],
            resolution=resolution,
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16.0, 16.0],
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--width", type=int, default=640)
    argparser.add_argument("--height", type=int, default=360)
    argparser.add_argument("--device-id", type=int, default=0)
    args = argparser.parse_args()
    device = get_device(args.device_id)

    # minerl.env.OrderedDict
    env_spec = ResizableObtainDiamondShovelEnvSpec(resolution=[args.width, args.height])
    env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = basalt_specs.BasaltTimeoutWrapper(env)
    env = basalt_specs.DoneOnESCWrapper(env)
    # env = gym.make("MineRLObtainDiamondShovel-v0")
    env.render_mode = "rgb_array"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="ksc-journal-performance",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="v1",
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env = VisionWrapper(env, args.width, args.height)
    env = SB3MineRLWrapper(env)
    vec_env = DummyVecEnv([lambda: Monitor(env)])
    # env = VecVideoRecorder(
    #     env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % 40000 == 0,
    #     video_length=20000,
    # )

    model = RecurrentPPO(
        "CnnLstmPolicy",
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}",
    )
    try:
        model.learn(
            total_timesteps=100000,
            callback=[
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
            ],
        )
    finally:
        vec_env.close()
        run.finish()

    # vec_env = make_vec_env(lambda: env, n_envs=1)
    # model = PPO("CnnPolicy", vec_env, verbose=1)

    # model.learn(total_timesteps=10000)

    # obs = env.reset()
    # done = False
    # while not done:
    #     ac = env.action_space.noop()
    #     # Spin around to see what is around us
    #     ac["camera"] = [0, 3]
    #     obs, reward, done, info = env.step(ac)
    #     env.render()
    # env.close()
