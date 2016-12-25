#!/usr/bin/python3
import os
import sys
import matplotlib.pyplot as plt
import gym
import numpy as np


class HillClimbing:

    def __init__(self):
        # input is sin(theta), cos(theta), and dtheta
        # we discretize these quantities as follows
        self.min_angle = -np.pi
        self.max_angle = np.pi
        self.angle_step = np.pi/4
        self.min_dtheta = -8
        self.max_dtheta = 8
        self.dtheta_step = 0.5
        self.min_action = -2
        self.max_action = 2
        self.action_step = 0.2
        self.angle_n = int((self.max_angle - self.min_angle) // self.angle_step + 1)
        self.dtheta_n = int((self.max_dtheta - self.min_dtheta) // self.dtheta_step + 1)
        self.action_n = int((self.max_action - self.min_action) // self.action_step + 1)
        self.q_size = self.angle_n * self.dtheta_n
        self.Q = np.zeros([self.q_size, self.action_n])

        self.lr = 0.4
        self.gamma = 0.9
        self.visited_states = []

    def compute_state_idx(self, observation):
        theta = np.arctan2(observation[1], observation[0])
        theta = theta - self.min_angle
        dtheta = observation[2] - self.min_dtheta
        theta //= self.angle_step
        dtheta //= self.dtheta_step

        # this is like flattening a 3d array
        return int(theta + (self.angle_n * dtheta))

    def compute_action_from_idx(self, action_idx):
        return action_idx * self.action_step + self.min_action

    def compute_action_idx(self, action):
        return int((action - self.min_action) // self.action_step)

    def run_episode(self, env, train_iter, render=False):
        observation = env.reset()
        total_reward = 0
        for _ in range(200):
            if render:
                env.render()

            state_idx = self.compute_state_idx(observation)

            if state_idx not in self.visited_states:
                self.visited_states.append(state_idx)

            if train_iter > 100:
                # greedily choose best action given q table, with some noise
                noisy_action_space = self.Q[state_idx] + np.random.randn(1, self.action_n) * 1
                action_idx = np.argmax(noisy_action_space)
            else:
                # pick random action
                action_idx = np.random.uniform(high=self.action_n)

            action = self.compute_action_from_idx(action_idx)

            # step the environment
            observation, reward, done, info = env.step([action])
            # cost has fixed lower limit, so we add to make it a reward
            reward += 16.273604

            # update Q table
            next_state_idx = self.compute_state_idx(observation)
            dq = reward + self.gamma * np.max(self.Q[next_state_idx])
            self.Q[state_idx, action_idx] = (1 - self.lr) * self.Q[state_idx, action_idx] + self.lr * dq
            total_reward += reward
            if done:
                break

        return total_reward

    def train(self, show_plot=False, upload=False):
        env = gym.make('Pendulum-v0')

        best_reward = -sys.maxsize

        # plotting stuff
        steps_between_plot = 200
        steps_between_render = 1000
        plt.ion()

        if upload:
            tag = '/tmp/' + os.path.basename(__file__) + '-' + str(int(np.random.rand() * 1000))
            env.monitor.start(tag)

        # simple Q learner
        rewards = []
        max_trials = 20000
        avg_reward = 0
        print('step, rewards, best_reward, 100_episode_avg_reward')
        for i in range(max_trials):

            reward = self.run_episode(env, i, render=False)
            rewards.append(reward)

            if show_plot and i % steps_between_plot == 0:
                plt.plot(rewards)
                plt.pause(0.05)

            print("%i, %d, %d, %f" % (i, reward, best_reward, avg_reward))
            if reward > best_reward:
                best_reward = reward

            if i > 100:
                rewards = rewards[-100:]
                avg_reward = sum(rewards) / 100.0
                if avg_reward > 5000.0:
                    print("game has been solved!")
                    break

        if upload:
            env.monitor.close()
            gym.upload(tag, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        np.savetxt('rewards.csv', rewards, delimiter=',')
        return best_reward


if __name__ == "__main__":
    hc = HillClimbing()
    r = hc.train(upload=False)
    print(r)
