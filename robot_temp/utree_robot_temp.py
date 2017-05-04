import matplotlib.pyplot as plt
import numpy
from continuous_utree import ContinuousUTree
import random
import gym

if __name__ == "__main__":
    numpy.seterr(all='raise')

    env = gym.make("RobotTemperature-v0")

    tree = ContinuousUTree(2, 6)

    # train
    rewards = []
    for i in range(1000):
        obs = env.reset()
        j = 0
        episode_reward = 0
        while True:
            move = int(random.random() + 0.5)
            temp = random.randint(0, 2)
            # env.render()
            action_idx = numpy.ravel_multi_index(([move, temp]), (2, 3))
            obs_prime, reward, done, info = env.step([move, temp])
            episode_reward += reward
            tree.add_transition(obs, action_idx, obs_prime, reward)
            # print(obs, reward)
            obs = obs_prime


            j += 1
            if j > 200:
                break

            if done:
                # print("success!")
                break

        did_split = tree.process()
        if did_split:
            print(tree.root)
        rewards.append(episode_reward)

    # test
    rewards = []
    for i in range(10):
        obs = env.reset()
        j = 0
        episode_reward = 0
        while True:
            env.render()

            action_idx = tree.best_action_idx(obs)
            move, temp = numpy.unravel_index(action_idx, (2, 3))
            obs_prime, reward, done, info = env.step([move, temp])
            tree.add_transition(obs, action_idx, obs_prime, reward)
            episode_reward += reward
            # print(obs, reward)
            obs = obs_prime

            j += 1
            if j > 200:
                break

            if done:
                tree.process()
                print("success!")
                break

        rewards.append(episode_reward)
        print(episode_reward)


    plt.plot(rewards)
    plt.show()
