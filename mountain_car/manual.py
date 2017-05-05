#!/usr/bin/python3
import sys
import time
import os
import matplotlib.pyplot as plt
from gym import wrappers
import gym
import numpy as np


class Manual:

    @staticmethod
    def run_episode(env, render=False):
        observation = env.reset()
        total_reward = 0
        for _ in range(400):
            if render:
                env.render()

            speed = observation[1]
            pos = observation[0]

            if speed < 0:
                action = 0
            else:
                action = 2

            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    @staticmethod
    def train(show_plot=False, upload=False):
        env = gym.make('MountainCar-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(directory)(env)
            env.monitored = True
        else:
            env.monitored = False

        reward = Manual.run_episode(env, render=True)
        print(reward)

if __name__ == "__main__":
    hc = Manual()
    hc.train(upload=False)
