import gym
import minerl

# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)

if __name__ == '__main__':
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    obs = env.reset()

    done = False
    while not done:
        ac = env.action_space.noop()
        # Spin around to see what is around us
        ac["camera"] = [0, 3]
        obs, reward, done, info = env.step(ac)
        env.render()
    env.close()
