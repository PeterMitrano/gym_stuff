#!/usr/bin/python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from math import cos, sin
import os
import sys
import gym
import numpy as np

def policy(observation):
    theta = np.arctan2(observation[1], observation[0])
    dtheta = observation[2]

    if abs(theta) > 0.2:  # gather momentum until we're close
        if (dtheta > 0 and dtheta < 3) and (theta > 0 or theta < -np.pi/2):
            action = 1.2
        elif (dtheta < 0 and dtheta > -3) and (theta < 0 or theta > np.pi/2):
            action = -1.2
        else:
            action = 0
    else:
        if theta > 0 and dtheta > 0:
            action = -2
        elif theta < 0 and dtheta < 0:
            action = 2
        else:
            action = 0

    return action

class ManualControl:

    @staticmethod
    def run_episode(env, train_iter, render=False):
        observation = env.reset()
        total_reward = 0
        rewards = []
        for _ in range(800):
            if render:
                env.render()

            action = policy(observation)

            # step the environment
            observation, reward, done, info = env.step([action])

            total_reward += reward
            rewards.append(reward)
            rewards = rewards[-20:]

            avg_reward = sum(rewards) / len(rewards)
            if avg_reward > -0.01:
                break

        return total_reward

    def train(self, show_plot=False, upload=False):
        env = gym.make('Pendulum-v0')

        best_reward = -sys.maxsize

        # plotting stuff
        steps_between_plot = 200
        plt.ion()

        if upload:
            tag = '/tmp/' + os.path.basename(__file__) + '-' + str(int(np.random.rand() * 1000))
            env.monitor.start(tag)

        # simple Q learner
        rewards = []
        max_trials = 200
        print_step = 1
        avg_reward = 0
        print('step, rewards, best_reward, 100_episode_avg_reward')
        for i in range(max_trials):

            reward = ManualControl.run_episode(env, i, render=True)
            rewards.append(reward)

            if show_plot and i % steps_between_plot == 0:
                plt.plot(rewards)
                plt.pause(0.05)

            if i % print_step == 0:
                print("%i, %d, %d, %f" % (i, reward, best_reward, avg_reward))

            if reward > best_reward:
                best_reward = reward

            last_100 = rewards[-100:]
            rewards = last_100
            avg_reward = sum(rewards) / len(last_100)

        if upload:
            env.monitor.close()
            gym.upload(tag, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        np.savetxt('rewards.csv', rewards, delimiter=',')


if __name__ == "__main__":
    hc = ManualControl()
    r = hc.train(upload=False)
