import numpy
from continuous_utree import ContinuousUTree
import random
import gym

if __name__ == "__main__":
    env = gym.make("RobotTemperature-v0")

    tree = ContinuousUTree(2, 6)

    for i in range(100):
        obs = env.reset()
        j = 0
        while True:
            move = int(random.random() + 0.5)
            temp = random.randint(0, 2)
            # env.render()
            obs_prime, reward, done, info = env.step([move, temp])
            action_idx = numpy.ravel_multi_index(([move, temp]), (2, 3))
            tree.add_transition(obs, action_idx, obs_prime, reward)
            # print(obs, reward)
            obs = obs_prime

            j += 1
            if j > 200:
                break

            if done:
                tree.process()
                print("success!")
                break
