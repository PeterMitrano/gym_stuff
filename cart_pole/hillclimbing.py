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
    def run_episode(env, parameters, render=False):
        observation = env.reset()
        total_reward = 0
        for _ in range(200):
            if render:
                env.render()
            # compute linear combination of parameters and observations
            if np.matmul(parameters, observation) < 0:
                action = 0
            else:
                action = 1
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
        env = gym.make('CartPole-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(directory)(env)
            env.monitored = True
        else:
            env.monitored = False

        # parameters start as random values between -1 and 1
        # EX of good params: [-0.1906  0.2306  0.4481  0.2017]
        observation_n = env.observation_space.shape[0]
        parameters = np.random.rand(observation_n) * 2 - 1
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
            noise_scaling = 1 - (i / max_trials)
            new_params = parameters + (np.random.rand(observation_n) * 2 - 1) * noise_scaling

            # normalize it because it reduced search space without changing action
            new_params = HillClimbing.normalize(new_params)
            reward = HillClimbing.run_episode(env, new_params)
            rewards.append(reward)

            if show_plot and i % steps_between_plot == 0:
                plt.plot(rewards)
                plt.pause(0.05)

            print("%i, %d, %d, %f" % (i, reward, best_reward, avg_reward))
            if reward > best_reward:
                best_reward = reward
                parameters = new_params

            if i > 100:
                avg_reward = sum(rewards[-100:]) / 100.0
                if avg_reward > 195.0:
                    print("game has been solved!")
                    break

        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        return parameters


if __name__ == "__main__":
    hc = HillClimbing()
    r = hc.train(upload=False)
    print(r)
