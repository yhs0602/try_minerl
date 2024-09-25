import crafter  # noqa
import gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from get_device import get_device
from sb3_wrapper import SB3Wrapper

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
        n_eval_episodes=3,
        deterministic=False,
        render=False,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}",
    )
    try:
        model.learn(
            total_timesteps=100000,
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
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        model.save(f"ckpts/v1-{run.name}.ckpt")
    finally:
        env.close()
        run.finish()
