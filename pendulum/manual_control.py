#!/usr/bin/python3
import time
import matplotlib.pyplot as plt
import os
import sys
import gym
import numpy as np


class ManualControl:

    @staticmethod
    def run_episode(env, train_iter, render=False):
        observation = env.reset()
        total_reward = 0
        for _ in range(400):
            if render:
                env.render()

            theta = np.arctan2(observation[1], observation[0])
            dtheta = observation[2]

            if abs(theta) > 0.2:  # gather momentum until we're close
                if (dtheta > 0 and dtheta < 4) and (theta > 0 or theta < -np.pi*2/3):
                        action = 1
                elif (dtheta < 0 and dtheta > -4) and (theta < 0 or theta > np.pi*2/3):
                    action = -1
                else:
                    action = 0
            else:
                if theta > 0 and dtheta > 0:
                    action = -2
                elif theta < 0 and dtheta < 0:
                    action = 2
                else:
                    action = 0

            time.sleep(0.1)

            # step the environment
            observation, reward, done, info = env.step([action])

            total_reward += reward
            if done:
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
        max_trials = 20000
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
            if avg_reward > -500.0:
                print("game has been solved!")
                break

        if upload:
            env.monitor.close()
            gym.upload(tag, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        np.savetxt('rewards.csv', rewards, delimiter=',')
        return best_reward


if __name__ == "__main__":
    hc = ManualControl()
    r = hc.train(upload=False)
    print(r)
