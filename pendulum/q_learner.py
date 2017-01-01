#!/usr/bin/python3
import bloscpack as bp
import time
import os
from math import sin, cos
import sys
import gym
from gym import wrappers
import numpy as np


class QLearner:

    def __init__(self):
        # input is sin(theta), cos(theta), and dtheta
        # we discretize these quantities as follows
        self.state_bounds = []
        self.state_bounds.append([-np.pi, np.pi, np.pi/24])
        self.state_bounds.append([-8, 8, 0.02])
        self.state_dims = []
        for bound in self.state_bounds:
            dim_n = (bound[1] - bound[0]) // bound[2] + 1
            self.state_dims.append(dim_n)

        self.action_bounds = [-2, 2, 0.2]
        self.action_n = (self.action_bounds[1] - self.action_bounds[0]) // self.action_bounds[2] + 1

        self.epsilon = 1e-8

        self.Q = np.random.rand(*self.state_dims, self.action_n)

        self.lr = 1.0  # optimal for deterministic environment
        self.gamma = 0.9
        self.visited_states = []

    def init_q_table(self, q_table):
        self.Q = q_table

    def compute_state_idx(self, observation):
        theta = np.arctan2(observation[1], observation[0])
        theta = theta - self.min_angle
        dtheta = observation[2] - self.min_dtheta
        theta = (theta) // self.angle_step
        dtheta = (dtheta) // self.dtheta_step

        # this is like flattening a 2d array
        return int(theta + (self.angle_n * dtheta))

    def compute_action_from_idx(self, action_idx):
        return action_idx * self.action_step + self.min_action

    def _compute_state_from_idx(self, state_idx):
        theta = (state_idx % self.angle_n) * self.angle_step + self.min_angle + self.epsilon
        if theta - self.epsilon <= self.max_angle < theta:
            theta -= self.epsilon
        dtheta = (state_idx // self.angle_n) * self.dtheta_step + self.min_dtheta + self.epsilon
        if dtheta - self.epsilon <= self.max_dtheta < dtheta:
            theta -= self.epsilon
        return [cos(theta), sin(theta), dtheta]

    def _compute_action_idx(self, action):
        if action > self.max_action:
            raise ValueError("Action value %f is greater than max of %f", action, self.max_action)

        action_idx = (action - self.min_action + self.epsilon) // self.action_step
        action_idx = min(action_idx, 19)
        return int(action_idx)

    def q_policy(self, observation, noise_level=1):
        state_idx = self.compute_state_idx(observation)
        # greedily choose best action given q table, with noise
        noise = np.random.rand(1, self.action_n) * noise_level
        action_idx = np.argmax(self.Q[state_idx] + noise)
        action = self.compute_action_from_idx(action_idx)
        return state_idx, action, action_idx

    def run_episode(self, env, noise_level, render=False):
        observation = env.reset()
        total_reward = 0

        i = 0
        while True:
            if render:
                env.render()

            state_idx, action, action_idx = self.q_policy(observation, noise_level=noise_level)

            # step the environment
            observation, reward, done, info = env.step([action])
            # cost has fixed lower limit, so we add to make it a reward

            # update Q table
            next_state_idx = self.compute_state_idx(observation)
            dq = reward + self.gamma * np.max(self.Q[next_state_idx])  # np.max() means use best action from next state
            new_q = (1 - self.lr) * self.Q[state_idx, action_idx] + self.lr * dq
            self.Q[state_idx, action_idx] = new_q
            total_reward += reward

            i += 1
            if env.monitored and done:
                break
            elif i > 600:
                break

        return total_reward

    def train(self, upload=False):
        env = gym.make('Pendulum-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(directory)(env)
            env.monitored = True
        else:
            env.monitored = False

        steps_between_render = 10000
        rewards = []
        max_trials = 100000
        print_step = 1000
        avg_reward = 0
        print('step, 100_episode_avg_reward')
        for i in range(max_trials):

            # decrease noise as we learn.
            noise_level = 36 / (12 + i)
            reward = self.run_episode(env, noise_level, render=False)

            if i % print_step == 0:
                rewards.append(reward)
                print("%i, %d" % (i, avg_reward))

            last_100 = rewards[-100:]
            rewards = last_100
            avg_reward = sum(rewards) / len(last_100)
            if avg_reward > -300.0 and len(last_100) == 100:
                print("game has been solved!")
                break

        # save q table
        bp.pack_ndarray_file(self.Q, 'q_table.bp')

        # upload/end monitoring
        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')


if __name__ == "__main__":
    ql = QLearner()
    ql.train(upload=False)
