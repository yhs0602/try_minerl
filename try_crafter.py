import sys
import time
import crafter
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from sb3_wrapper import SB3Wrapper

if __name__ == '__main__':
    crafter
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="crafter-test",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="v1",
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    env = gym.make('CrafterReward-v1')  # Or CrafterNoReward-v1
    # env = crafter.Recorder(
    #     env, './path/to/logdir',
    #     save_stats=True,
    #     save_video=True,
    #     save_episode=False,
    # )
    env.render_mode = 'rgb_array'
    env = Monitor(SB3Wrapper(env))
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 40000 == 0,
        video_length=20000,
    )

    obs = env.reset()
    start_time = time.time_ns()
    done = False
    for i in range(9000000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action])
        time_elapsed = max((time.time_ns() - start_time) / 1e9, sys.float_info.epsilon)
        fps = int(i / time_elapsed)
        if i % 64 == 0:
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
        if done:
            break
    env.close()
    run.finish()
