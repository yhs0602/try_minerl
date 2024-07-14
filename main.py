import sys
import time

import gym
import minerl
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from vision_wrapper import VisionWrapper

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

if __name__ == '__main__':
    minerl.env.OrderedDict
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    env.render_mode = "rgb_array"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="minerl-test",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="v1",
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env = Monitor(VisionWrapper(env, 640, 360))
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 40000 == 0,
        video_length=20000,
    )

    vec_env = env
    obs = vec_env.reset()
    start_time = time.time_ns()
    for i in range(9000000):
        # sample one from the action space
        ac = env.action_space.noop()
        # print(f"Action: {action}")
        obs, reward, done, info = vec_env.step([ac])
        time_elapsed = max(
            (time.time_ns() - start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int(i / time_elapsed)
        if i % 512 == 0:
            wandb.log(
                {
                    "time/iterations": i,
                    "time/fps": fps,
                    "time/time_elapsed": int(time_elapsed),
                    "time/total_timesteps": i,
                }
            )
        if i % 4000 == 0:
            print(f"Step: {i}")
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
