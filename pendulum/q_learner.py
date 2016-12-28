#!/usr/bin/python3
import time
import pickle
import os
import manual_control
from math import sin, cos
import sys
import matplotlib.pyplot as plt
import gym
from gym import wrappers
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

    def random_policy(self, observation):
        state_idx = self.compute_state_idx(observation)
        action_idx = np.random.randint(0, self.action_n)
        action = self.compute_action_from_idx(action_idx)
        return state_idx, action, action_idx

    def manual_policy(self, observation):
        state_idx = self.compute_state_idx(observation)
        action = manual_control.policy(observation)
        action_idx = self.compute_action_idx(action)
        return state_idx, action, action_idx

    def noisy_q_policy(self, observation):
        state_idx = self.compute_state_idx(observation)
        # greedily choose best action given q table, with noise
        noise = np.random.rand(1, self.action_n) * 5
        action_idx = np.argmax(self.Q[state_idx] + noise)
        action = self.compute_action_from_idx(action_idx)
        return state_idx, action, action_idx

    def run_episode(self, env, training, render=False):
        observation = env.reset()
        total_reward = 0
        rewards = []
        for _ in range(600):
            if render:
                env.render()

            # Q Learning is off policy, so we can follow a much better (manual) policy while we learn
            state_idx, action, action_idx = self.noisy_q_policy(observation)

            # step the environment
            observation, reward, done, info = env.step([action])
            # cost has fixed lower limit, so we add to make it a reward

            # update Q table
            next_state_idx = self.compute_state_idx(observation)
            dq = reward + self.gamma * np.max(self.Q[next_state_idx])  # np.max() means use best action from next state
            new_q = (1 - self.lr) * self.Q[state_idx, action_idx] + self.lr * dq
            self.Q[state_idx, action_idx] = new_q
            total_reward += reward

            rewards.append(reward)
            rewards = rewards[-80:]

            avg_reward = sum(rewards) / len(rewards)
            if avg_reward > -0.005 or done:
                break

        return total_reward

    def train(self, show_plot=False, upload=False):
        env = gym.make('Pendulum-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(directory)(env)

        best_reward = -sys.maxsize

        steps_between_render = 1000
        rewards = []
        max_trials = 100000
        print_step = 500
        avg_reward = 0
        print('step, rewards, best_reward, 100_episode_avg_reward')
        for i in range(max_trials):

            if i % steps_between_render == 0:
                reward = self.run_episode(env, i, render=True)
            else:
                reward = self.run_episode(env, i, render=False)

            if i % print_step == 0:
                rewards.append(reward)
                print("%i, %d, %d, %f" % (i, reward, best_reward, avg_reward))

            if reward > best_reward:
                best_reward = reward

            last_100 = rewards[-100:]
            rewards = last_100
            avg_reward = sum(rewards) / len(last_100)
            if avg_reward > -300.0:
                print("game has been solved!")
                break

        plt.plot(rewards)
        plt.show()

        # save q table
        pickle.dump(self.Q, open('q_table.pickle', 'wb'))

        # upload/end monitoring
        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')

        return best_reward


if __name__ == "__main__":
    ql = QLearner()
    ql.init_q_from_manual_policy()
    r = ql.train(show_plot=True, upload=False)
    # print(r)
