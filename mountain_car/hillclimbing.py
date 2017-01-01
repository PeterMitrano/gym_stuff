#!/usr/bin/python3
import sys
import time
import os
import matplotlib.pyplot as plt
from gym import wrappers
import gym
import numpy as np


class HillClimbing:

    @staticmethod
    def run_episode(env, weights, biases, render=False):
        observation = env.reset()
        total_reward = 0
        avg_action = 0
        for _ in range(400):
            if render:
                env.render()
            # compute linear combination of parameters and observations
            p = sum(weights * observation + biases)
            if p < 0:
                action = 0
            else:
                action = 2
            avg_action += action
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    @staticmethod
    def normalize(x):
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        else:
            return x/norm

    @staticmethod
    def train(show_plot=False, upload=False):
        env = gym.make('MountainCar-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(directory)(env)
            env.monitored = True
        else:
            env.monitored = False

        # parameters start as random values between -1 and 1
        observation_n = env.observation_space.shape[0]
        weights = np.zeros(observation_n)
        biases = np.zeros(observation_n)
        best_reward = -sys.maxsize

        # plotting stuff
        steps_between_plot = 40
        plt.ion()

        # 2000 episodes
        rewards = []
        max_trials = 2000
        avg_reward = 0
        print('step, rewards, best_reward, 100_episode_avg_reward')
        for i in range(max_trials):
            noise_scaling = 20 / (200 + i)
            new_weights = weights + (np.random.rand(observation_n) * 2 - 1) * noise_scaling
            new_biases = biases + (np.random.rand(observation_n) * 2 - 1) * noise_scaling

            # normalize it because it reduced search space without changing action
            # new_weights = HillClimbing.normalize(new_weights)
            # new_biases = HillClimbing.normalize(new_biases)

            sample_reward = 0
            for j in range(100):
                if i % 200 == 0 and j == 0:
                    reward = HillClimbing.run_episode(env, new_weights, new_biases, render=True)
                else:
                    reward = HillClimbing.run_episode(env, new_weights, new_biases, render=False)
                sample_reward += reward
            rewards.append(sample_reward)

            if show_plot and i % steps_between_plot == 0:
                plt.plot(rewards)
                plt.pause(0.05)

            if reward > best_reward:
                best_reward = reward
                weights = new_weights
                biases = new_biases

            avg_reward = sample_reward / 100.0
            print("%i, %d" % (i, avg_reward))


        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        return weights


if __name__ == "__main__":
    hc = HillClimbing()
    r = hc.train(upload=False)
    print(r)
