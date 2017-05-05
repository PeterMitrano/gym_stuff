#!/usr/bin/python3
from datetime import datetime
import tensorflow as tf
import numpy as np
import time
import os
from gym import wrappers
import gym


class PolicyInModel:
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
                policy_vars = [policy_w1, policy_b1, policy_w2, policy_b2]

        with tf.name_scope('model'):
            with tf.name_scope('h1'):
                model_input = tf.concat((policy_action, state), axis=1, name='concat')

                model_h1_dim = 10
                model_w1 = tf.Variable(tf.truncated_normal([state_dim + action_dim, model_h1_dim], 0, 0.1),
                                       name='model_w1')
                model_b1 = tf.Variable(tf.constant(0.1, shape=[model_h1_dim]), name='model_b1')
                model_h1 = tf.nn.relu(tf.matmul(model_input, model_w1) + model_b1)

                model_w2 = tf.Variable(tf.truncated_normal((model_h1_dim, state_dim), 0, 0.1), name='model_w2')
                model_b2 = tf.Variable(tf.constant(0.1, shape=[state_dim]), name='model_b2')
                model_raw_state = tf.nn.sigmoid(tf.matmul(model_h1, model_w2) + model_b2, name='model_norm')
                next_state = model_raw_state * (state_bounds[:, 1] - state_bounds[:, 0]) + state_bounds[:, 0]
                model_vars = [model_w1, model_b1, model_w2, model_b2]

        with tf.name_scope("loss"):
            policy_loss = tf.nn.l2_loss(next_state - state, name='policy_loss')
            model_loss = tf.nn.l2_loss(next_state - true_next_state, name='model_loss')
            tf.summary.scalar("policy_loss", policy_loss)
            tf.summary.scalar("model_loss", model_loss)

        learning_rate = 0.001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_policy = optimizer.minimize(policy_loss, var_list=policy_vars)
        train_model = optimizer.minimize(model_loss, var_list=model_vars)

        init = tf.global_variables_initializer()

        merged_summary = tf.summary.merge_all()

        stamp = "{:%B_%d_%H:%M:%S}".format(datetime.now())
        tag = input("description of this run (or leave blank):")
        log_dir = 'log_data/' + stamp + "/"
        meta_file = open('')
        writer = tf.summary.FileWriter(log_dir)

        with tf.Session() as sess:

            sess.run(init)

            while True:
                episode_iters = 0
                observation = env.reset()
                action = np.random.randint(0, 3)
                next_observation = env.step(action)[0]

                while episode_iters < 400:

                    _, action, m_loss = sess.run([train_model, policy_action, model_loss],
                                                 feed_dict={state: [observation], true_next_state: [next_observation]})
                    _ = sess.run([train_policy], feed_dict={state: [observation], true_next_state: [next_observation]})

                    if episode_iters % 50 == 0:
                        summary = sess.run([merged_summary])

                        print(m_loss)
                    # action = np.argmax(action)
                    # print(action)

                    # OFF POLICY LEARNING FOR NOW
                    speed = observation[1]
                    pos = observation[0]

                    if speed < 0:
                        action = 0
                    else:
                        action = 2

                    observation = next_observation
                    next_observation, reward, done, info = env.step(action)
                    episode_iters += 1

                    # env.render()

                    if done:
                        break

        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')


if __name__ == "__main__":
    hc = PolicyInModel()
    hc.main(upload=False)
