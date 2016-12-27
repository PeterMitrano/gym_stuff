#!/usr/bin/python3
from mpl_toolkits.mplot3d import Axes3D
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from math import cos, sin
import os
import sys
import gym
import numpy as np
import manual_control


class NNControl:

    @staticmethod
    def run_episode(sess, env, action_pred, X, render=False):
        observation = env.reset()
        total_reward = 0
        rewards = []
        for _ in range(800):
            if render:
                env.render()

            observation = np.reshape(observation, (1, 3))
            action = sess.run(action_pred, feed_dict={X: observation})

            # step the environment
            observation, reward, done, info = env.step([action])

            total_reward += reward
            rewards.append(reward)

            avg_reward = sum(rewards) / len(rewards)
            if avg_reward > -0.01:
                break

        return total_reward

    @staticmethod
    def run(sess, action_pred, X, upload=False):
        env = gym.make('Pendulum-v0')

        best_reward = -sys.maxsize

        tag = ''
        if upload:
            tag = '/tmp/' + os.path.basename(__file__) + '-' + str(int(np.random.rand() * 1000))
            env.monitor.start(tag)

        rewards = []
        max_trials = 2000
        print_step = 1
        avg_reward = 0
        print('step, rewards, best_reward, 100_episode_avg_reward')
        for i in range(max_trials):

            reward = NNControl.run_episode(sess, env, action_pred, X, render=True)
            rewards.append(reward)

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

    @staticmethod
    def test(sess, training_costs, action_pred, X):
        xs = []
        ys = []
        zs = []
        pred_zs = []
        for _ in range(2000):
            theta = np.random.uniform(-np.pi, np.pi)
            dtheta = np.random.uniform(-8, 8)
            obs = [cos(theta), sin(theta), dtheta]
            action = manual_control.policy(obs)
            obs_arr = np.array([obs])

            xs.append(theta)
            ys.append(dtheta)

            pred_z = sess.run(action_pred, feed_dict={X: obs_arr})[0][0]

            pred_zs.append(pred_z)
            zs.append(action)

        action_fig = plt.figure(1)
        training_fig = plt.figure(2)
        training_ax = training_fig.add_subplot(111)
        training_ax.plot(training_costs)
        ax = action_fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs=pred_zs, c='b', label='pred')
        ax.scatter(xs, ys, zs=zs, c='r', label='true')
        plt.show()

def main(upload=False):
    n_hidden_1 = 16
    n_hidden_2 = 16

    with tf.name_scope("input"):
        X = tf.placeholder(np.float32, shape=[1, 3])
        W1 = tf.Variable(np.random.normal(size=[3, n_hidden_1]), dtype=np.float32)
        b1 = tf.Variable(np.random.normal(size=[1, n_hidden_1]), dtype=np.float32)
        fc1 = tf.matmul(X, W1)
        fc1 = tf.add(fc1, b1)
        fc1 = tf.nn.sigmoid(fc1)

    with tf.name_scope("h1"):
        W2 = tf.Variable(np.random.normal(size=[n_hidden_1, n_hidden_2]), dtype=np.float32)
        b2 = tf.Variable(np.random.normal(size=[1, n_hidden_2]), dtype=np.float32)
        fc2 = tf.matmul(fc1, W2)
        fc2 = tf.add(fc2, b2)
        fc2 = tf.nn.sigmoid(fc2)

    with tf.name_scope("h2"):
        W3 = tf.Variable(np.random.normal(size=[n_hidden_2, 1]), dtype=np.float32)
        b3 = tf.Variable(np.random.normal(size=[1, 1]), dtype=np.float32)
        fc3 = tf.matmul(fc2, W3)
        fc3 = tf.add(fc3, b3)

    action_pred = fc3
    action_true = tf.placeholder(np.float32, shape=[1, None])

    with tf.name_scope("error"):
        error = action_pred - action_true
        # tf.scalar_summary("error", error)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.pow(error, 2))
        tf.scalar_summary("cost", cost)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as sess:
        writer = tf.train.SummaryWriter(".logs", sess.graph)  # for 0.8
        summaries = tf.merge_all_summaries()

        sess.run(tf.initialize_all_variables())

        # the training phase
        n = 20000
        avg_costs = []
        print("step, cost")
        for i in range(20):
            costs = 0
            for _ in range(n):
                theta = np.random.uniform(-np.pi, np.pi)
                dtheta = np.random.uniform(-8, 8)
                obs = [cos(theta), sin(theta), dtheta]
                action = manual_control.policy(obs)
                obs_arr = np.array([obs])
                action_arr = np.array([[action]])

                s, z, c, _ = sess.run([summaries, action_pred, cost, optimizer], feed_dict={X: obs_arr, action_true: action_arr})
                z = z[0, 0]
                costs += c
                writer.add_summary(s)

            avg_cost = costs/n
            print(i, avg_cost)
            avg_costs.append(costs/n)

        print("Training is complete.")

        nnc = NNControl()
        # nnc.test(sess, avg_costs, action_pred, X)
        nnc.run(sess, action_pred, X)

if __name__ == "__main__":
    main()
