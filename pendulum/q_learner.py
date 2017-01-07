#!/usr/bin/python3
from math import sin, cos
import numpy as np


class QLearner:

    def __init__(self, state_bounds, action_bounds):
        # input is sin(theta), cos(theta), and dtheta
        # we discretize these quantities as follows
        self.state_bounds = state_bounds
        self.state_dims = []
        for bound in self.state_bounds:
            dim_n = (bound[1] - bound[0]) // bound[2] + 1
            self.state_dims.append(dim_n)

        self.action_bounds = action_bounds
        self.action_dims = []
        for bound in self.action_bounds:
            dim_n = (bound[1] - bound[0]) // bound[2] + 1
            self.action_dims.append(dim_n)

        self.epsilon = 1e-8

        self.Q = np.random.rand(*self.state_dims, *self.action_dims)

        self.lr = 1.0  # optimal for deterministic environment
        self.gamma = 0.9
        self.visited_states = []

    def init_q_table(self, q_table):
        self.Q = q_table

    def compute_state_idx(self, observation):
        idxs = []
        for dim in observation:
            min_obs = self.state_bounds[0][0]
            step_obs = self.state_bounds[0][2]
            idxs.append((dim - min_obs) // step_obs)

        state_idx = np.ravel_multi_index(idxs, tuple(self.state_dims))
        return state_idx

    def compute_action_from_idx(self, action_idx):
        idxs = np.unravel_index(action_idx, tuple(self.action_dims))
        action = []
        for idx in idxs:
            min_act = self.action_bounds[0][0]
            step_act = self.action_bounds[0][2]
            act = idx * step_act + min_act
            action.append(act)

        return action

    def _compute_state_from_idx(self, state_idx):
        idxs = np.unravel_index(state_idx, tuple(self.state_dims))
        state = []
        for idx in idxs:
            min_st = self.state_bounds[0][0]
            step_st = self.state_bounds[0][2]
            st = idx * step_st + min_st
            state.append(st)

        return state

    def _compute_action_idx(self, action):
        idxs = []
        for dim in action:
            min_obs = self.action_bounds[0][0]
            step_obs = self.action_bounds[0][2]
            idxs.append((dim - min_obs) // step_obs)

        action_idx = np.ravel_multi_index(idxs, tuple(self.action_dims))
        return action_idx

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
