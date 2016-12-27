#!/usr/bin/python3
import os
import time
import manual_control
from math import sin, cos
import sys
import matplotlib.pyplot as plt
import gym
import numpy as np


class QLearner:

    def __init__(self):
        # input is sin(theta), cos(theta), and dtheta
        # we discretize these quantities as follows
        self.epsilon = 1e-8
        self.min_angle = -np.pi
        self.max_angle = np.pi
        self.angle_step = np.pi/16
        self.min_dtheta = -8
        self.max_dtheta = 8
        self.dtheta_step = 0.05
        self.min_action = -2
        self.max_action = 2
        self.action_step = 0.2
        self.angle_n = int((self.max_angle - self.min_angle) // self.angle_step + 1)
        self.dtheta_n = int((self.max_dtheta - self.min_dtheta) // self.dtheta_step + 1)
        self.action_n = int((self.max_action - self.min_action) // self.action_step + 1)
        self.states_n = self.angle_n * self.dtheta_n
        self.Q = np.zeros([self.states_n, self.action_n])

        self.lr = 0.4
        self.gamma = 0.9
        self.visited_states = []

    def init_q_from_manual_policy(self):
        initial_reward = 10
        for state_idx in range(self.states_n):
            state = self.compute_state_from_idx(state_idx)
            action = manual_control.policy(state)
            action_idx = self.compute_action_idx(action)
            self.Q[state_idx, action_idx] = initial_reward

    def compute_state_idx(self, observation):
        theta = np.arctan2(observation[1], observation[0])
        theta = theta - self.min_angle
        dtheta = observation[2] - self.min_dtheta
        theta = (theta) // self.angle_step
        dtheta = (dtheta) // self.dtheta_step

        # this is like flattening a 2d array
        return int(theta + (self.angle_n * dtheta))

    def compute_state_from_idx(self, state_idx):
        theta = (state_idx % self.angle_n) * self.angle_step + self.min_angle + self.epsilon
        if theta - self.epsilon <= self.max_angle < theta:
            theta -= self.epsilon
        dtheta = (state_idx // self.angle_n) * self.dtheta_step + self.min_dtheta + self.epsilon
        if dtheta - self.epsilon <= self.max_dtheta < dtheta:
            theta -= self.epsilon
        return [cos(theta), sin(theta), dtheta]

    def compute_action_from_idx(self, action_idx):
        return action_idx * self.action_step + self.min_action

    def compute_action_idx(self, action):
        if action > self.max_action:
            raise ValueError("Action value %f is greater than max of %f", action, self.max_action)

        action_idx = (action - self.min_action + self.epsilon) // self.action_step
        action_idx = min(action_idx, 19)
        return int(action_idx)

    def policy(self, observation):
        state_idx = self.compute_state_idx(observation)
        # greedily choose best action given q table, with NO noise
        action_idx = np.argmax(self.Q[state_idx])
        action = self.compute_action_from_idx(action_idx)
        return action

    def run_episode(self, env, train_iter, render=False):
        observation = env.reset()
        total_reward = 0
        rewards = []
        for _ in range(600):
            if render:
                env.render()
                time.sleep(0.1)

            action = self.policy(observation)

            # step the environment
            observation, reward, done, info = env.step([action])
            # cost has fixed lower limit, so we add to make it a reward

            # update Q table
            # next_state_idx = self.compute_state_idx(observation)
            # dq = reward + self.gamma * np.max(self.Q[next_state_idx])  # np.max() means use best action from next state
            # self.Q[state_idx, action_idx] = (1 - self.lr) * self.Q[state_idx, action_idx] + self.lr * dq
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
        steps_between_render = 1000
        plt.ion()

        if upload:
            tag = '/tmp/' + os.path.basename(__file__) + '-' + str(int(np.random.rand() * 1000))
            env.monitor.start(tag)

        # simple Q learner
        rewards = []
        max_trials = 20000
        print_step = 100
        avg_reward = 0
        print('step, rewards, best_reward, 100_episode_avg_reward')
        for i in range(max_trials):

            reward = self.run_episode(env, i, render=True)
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
            if avg_reward > 5000.0:
                print("game has been solved!")
                break

        if upload:
            env.monitor.close()
            gym.upload(tag, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        np.savetxt('rewards.csv', rewards, delimiter=',')
        return best_reward


if __name__ == "__main__":
    ql = QLearner()
    ql.init_q_from_manual_policy()

    # xs = []
    # ys = []
    # zs = []
    # pred_xs = []
    # pred_ys = []
    # pred_zs = []
    # for state_idx in range(hc.states_n):
    #     state = hc.compute_state_from_idx(state_idx)
    #     theta = np.arctan2(state[1], state[0])
    #     dtheta = state[2]
    #     action_idx = np.argmax(hc.Q[state_idx])
    #     action = hc.compute_action_from_idx(action_idx)
    #
    #     pred_xs.append(theta)
    #     pred_ys.append(dtheta)
    #     pred_zs.append(action)
    #
    #     theta = np.random.uniform(-np.pi, np.pi)
    #     dtheta = np.random.uniform(-8, 8)
    #     obs = [cos(theta), sin(theta), dtheta]
    #     action = manual_control.policy(obs)
    #     xs.append(theta)
    #     ys.append(dtheta)
    #     zs.append(action)
    #
    # action_fig = plt.figure(1)
    # action_fig2 = plt.figure(2)
    # ax = action_fig.add_subplot(111, projection='3d')
    # ax2 = action_fig2.add_subplot(111, projection='3d')
    # ax.scatter(pred_xs, pred_ys, zs=pred_zs, c='b', label='pred')
    # ax2.scatter(xs, ys, zs=zs, c='r', label='true')
    # plt.show()

    for i in range(ql.states_n):
        x = ql.compute_state_from_idx(i)
        i2 = ql.compute_state_idx(x)
        if i != i2:
            print(i, "!=", i2)
            break

    # r = ql.train(upload=False)
    # print(r)
