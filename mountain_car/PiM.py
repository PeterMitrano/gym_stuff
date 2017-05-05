#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import time
import os
from gym import wrappers
import gym


class PolicyInModel:
    @staticmethod
    def run_episode(env, render=False):
        observation = env.reset()
        total_reward = 0
        for _ in range(400):
            if render:
                env.render()

            speed = observation[1]

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
    def main(upload=False):
        env = gym.make('MountainCar-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(directory)(env)
            env.monitored = True
        else:
            env.monitored = False

        state_dim = 2
        state_bounds = np.array([[-1.2, 0.6], [-0.07, 0.07]])
        action_dim = 3
        state = tf.placeholder(tf.float32, shape=[1, state_dim], name='state')
        true_next_state = tf.placeholder(tf.float32, shape=[1, state_dim], name='state_next')

        with tf.name_scope('policy'):
            with tf.name_scope('h1'):
                policy_h1_dim = 20
                policy_w1 = tf.Variable(tf.truncated_normal([state_dim, policy_h1_dim], 0, 0.1), name='policy_w1')
                policy_b1 = tf.Variable(tf.constant(0.1, shape=[policy_h1_dim]), name='policy_b1')
                policy_h1 = tf.nn.relu(tf.matmul(state, policy_w1, name='matmul1') + policy_b1, name='relu')

                policy_w2 = tf.Variable(tf.truncated_normal((policy_h1_dim, action_dim), 0, 0.1), name='policy_w2')
                policy_b2 = tf.Variable(tf.constant(0.1, shape=[action_dim]), name='policy_b2')
                policy_action = tf.nn.softmax(tf.matmul(policy_h1, policy_w2, name='matmul1') + policy_b2)

        with tf.name_scope('model'):
            with tf.name_scope('h1'):
                model_h1_dim = 10
                model_w1 = tf.Variable(tf.truncated_normal([action_dim, model_h1_dim], 0, 0.1), name='model_w1')
                model_b1 = tf.Variable(tf.constant(0.1, shape=[model_h1_dim]), name='model_b1')
                model_h1 = tf.nn.relu(tf.matmul(policy_action, model_w1) + model_b1)

                model_w2 = tf.Variable(tf.truncated_normal((model_h1_dim, state_dim), 0, 0.1), name='model_w2')
                model_b2 = tf.Variable(tf.constant(0.1, shape=[state_dim]), name='model_b2')
                model_raw_state = tf.nn.sigmoid(tf.matmul(model_h1, model_w2) + model_b2, name='model_norm')
                next_state = model_raw_state * (state_bounds[:, 1] - state_bounds[:, 0]) + state_bounds[:, 0]

                policy_loss = tf.nn.l2_loss(next_state - state, name='policy_loss')
                model_loss = tf.nn.l2_loss(next_state - true_next_state, name='model_loss')

        

        # reward = Manual.run_episode(env, render=True)
        # print(reward)

        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')


if __name__ == "__main__":
    hc = PolicyInModel()
    hc.main(upload=False)
