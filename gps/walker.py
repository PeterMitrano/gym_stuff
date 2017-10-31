import gym
import sys


def main():
    env = gym.make("Walker2d-v1")
    obs = env.reset()
    print(env.action_space)

    for i in range(10):
        obs, reward, done, info = env.step([0, 0])
        env.render()


if __name__ == "__main__":
    sys.exit(main())