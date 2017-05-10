#!/usr/bin/python3

import os
import sys
import time
from datetime import datetime
from subprocess import call

import numpy as np
import tensorflow as tf

import gym
from gym import wrappers


class PolicyInModel:
    def __init__(self):
        self.state_dim = 2
        self.state_bounds = np.array([[-1.2, 0.6], [-0.07, 0.07]])
        self.state_sizes = self.state_bounds[:, 1] - self.state_bounds[:, 0]
        self.action_dim = 3
        # self.on_policy_learning = True
        self.on_policy_learning = False

        with tf.name_scope("fed_values"):
            self.state = tf.placeholder(tf.float32, shape=[1, self.state_dim], name='state')
            self.true_next_state = tf.placeholder(tf.float32, shape=[1, self.state_dim], name='state_next')
            self.manual_action = tf.placeholder(tf.int32, shape=[], name='manual_action')
            self.reward = tf.placeholder(tf.int32, shape=[], name='reward')
            tf.summary.scalar("true_next_position", self.true_next_state[0][0])
            tf.summary.scalar("true_next_velocity", self.true_next_state[0][1])
            tf.summary.scalar("reward", self.reward)

        if self.on_policy_learning:
            with tf.name_scope('policy'):
                # self.policy_h1_dim = 5
                # self.policy_w1 = tf.Variable(tf.truncated_normal([self.state_dim, self.policy_h1_dim], 0, 0.1),
                #                              name='policy_w1')
                # self.policy_b1 = tf.Variable(tf.constant(0.1, shape=[self.policy_h1_dim]), name='policy_b1')
                # self.policy_h1 = tf.nn.tanh(tf.matmul(self.state, self.policy_w1, name='matmul1') + self.policy_b1,
                #                             name='tanh')
                #
                # self.policy_w2 = tf.Variable(tf.truncated_normal((self.policy_h1_dim, self.action_dim), 0, 0.1),
                #                              name='policy_w2')
                # self.policy_b2 = tf.Variable(tf.constant(0.1, shape=[self.action_dim]), name='policy_b2')
                # self.policy_action_float = tf.nn.softmax(tf.matmul(self.policy_h1, self.policy_w2, name='matmul1') + self.policy_b2)

                self.policy_w1 = tf.Variable(tf.truncated_normal([self.state_dim, self.action_dim], 0, 0.1), name='policy_w1')
                self.policy_action_float = tf.matmul(self.state, self.policy_w1, name='matmul1')
                self.gumbel = -tf.log(-tf.log(tf.random_uniform([], 0, 1, tf.float32)), name='gumbel')
                self.policy_temp = 0.1
                self.policy_softmax = tf.nn.softmax((self.policy_action_float + self.gumbel) / self.policy_temp)
                self.policy_action = tf.argmax(self.policy_action_float, axis=1)[0]
                self.policy_vars = [self.policy_w1]

                tf.summary.histogram('policy_w1', self.policy_w1)
                # tf.summary.histogram('policy_b1', self.policy_b1)
                # tf.summary.histogram('policy_w2', self.policy_w2)
                # tf.summary.histogram('policy_b2', self.policy_b2)
                tf.summary.histogram('policy_softmax', self.policy_softmax)

        with tf.name_scope("action"):
            if self.on_policy_learning:
                tf.summary.scalar("policy_action", self.policy_action)
            else:
                tf.summary.scalar("manual_action", self.manual_action)

        with tf.name_scope('model'):
            if self.on_policy_learning:
                self.action_input = self.policy_softmax
            else:
                action_in = tf.one_hot(self.manual_action, self.action_dim, dtype=tf.float32, name='manual_action_float')
                self.action_input = tf.expand_dims(action_in, axis=0)
            # hard coded shit
            # self.model_w1 = tf.Variable([[1.00, -0.0025], [1, 0], [-0.001, 0], [0, 0], [0.001, 0]], name='model_w1')
            # self.predicted_next_state = tf.matmul(self.model_input, self.model_w1)
            # self.model_vars = [self.model_w1]

            # fancy model
            with tf.name_scope("model_state"):
                self.model_state_dim = 4
                self.model_w_state = tf.Variable(tf.truncated_normal([self.state_dim, self.model_state_dim], 0, 0.1), name='model_w_state')
                self.model_b_state = tf.Variable(tf.constant(0.1, shape=[self.model_state_dim]), name='model_b_state')
                self.model_state_h1 = tf.nn.tanh(tf.matmul(self.state / self.state_sizes, self.model_w_state) + self.model_b_state)

            with tf.name_scope("model_action"):
                self.model_action_dim = 4
                self.model_w_action = tf.Variable(tf.truncated_normal([self.action_dim, self.model_action_dim], 0, 0.1), name='model_w_action')
                self.model_b_action = tf.Variable(tf.constant(0.1, shape=[self.model_action_dim]), name='model_b_action')
                self.model_action_h1 = tf.nn.tanh(tf.matmul(self.action_input, self.model_w_action) + self.model_b_action)

            with tf.name_scope("model_h2"):
                self.model_h1 = tf.concat((self.model_action_h1, self.model_state_h1), axis=1)
                self.model_w2 = tf.Variable(tf.truncated_normal([self.model_action_dim + self.model_state_dim, self.state_dim], 0, 0.1), name='model_w2')
                self.model_b2 = tf.Variable(tf.constant(0.1, shape=[self.state_dim]), name='model_b2')
                self.unadjusted_next_state = tf.matmul(self.model_h1, self.model_w2) + self.model_b2
                self.predicted_next_state = tf.nn.sigmoid(self.unadjusted_next_state) * self.state_sizes + self.state_bounds[:, 0]

            self.model_vars = [self.model_w_state, self.model_w_action, self.model_b_state, self.model_b_action, self.model_w2, self.model_b2]

            tf.summary.histogram("model_w_state", self.model_w_state)
            tf.summary.histogram("model_w_action", self.model_w_action)
            tf.summary.histogram("model_b_state", self.model_b_state)
            tf.summary.histogram("model_b)action", self.model_b_action)
            tf.summary.histogram("model_w2", self.model_w2)
            tf.summary.histogram("model_b2", self.model_b2)
            tf.summary.scalar("predicted_position", self.predicted_next_state[0][0])
            tf.summary.scalar("predicted_velocity", self.predicted_next_state[0][1])

        with tf.name_scope("loss"):
            self.state_change = self.predicted_next_state - self.state
            self.policy_loss = -tf.nn.l2_loss(self.state_change, name='policy_loss')
            # self.policy_loss = 0.5 - self.predicted_next_state[0][0]
            self.model_loss = tf.nn.l2_loss((self.predicted_next_state - self.true_next_state), name='model_loss')

            tf.summary.scalar("policy_loss", self.policy_loss)
            tf.summary.scalar("model_loss", self.model_loss)

        if self.on_policy_learning:
            with tf.name_scope("policy_gradients"):
                grads = list(zip(tf.gradients(self.policy_loss, self.policy_vars), self.policy_vars))
                for grad, var in grads:
                    tf.summary.histogram(var.name + '/gradient', grad)

        with tf.name_scope("model_gradients"):
            grads = list(zip(tf.gradients(self.model_loss, self.model_vars), self.model_vars))
        for grad, var in grads:
            tf.summary.histogram(var.name + '/gradient', grad)

        self.global_step = tf.Variable(0, trainable=False)
        self.model_optimizer = tf.train.GradientDescentOptimizer(1)
        self.policy_optimizer = tf.train.GradientDescentOptimizer(0.0001)
        self.train_model = self.model_optimizer.minimize(self.model_loss, var_list=self.model_vars,
                                                         global_step=self.global_step)
        if self.on_policy_learning:
            self.train_policy = self.policy_optimizer.minimize(self.policy_loss, var_list=self.policy_vars,
                                                               global_step=self.global_step)

        self.init = tf.global_variables_initializer()

        self.merged_summary = tf.summary.merge_all()

    def main(self, upload=False):
        env = gym.make('MountainCar-v0')
        directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
        if upload:
            env = wrappers.Monitor(env, directory)
            env.monitored = True
        else:
            env.monitored = False

        day_str = "{:%B_%d}".format(datetime.now())
        time_str = "{:%H:%M:%S}".format(datetime.now())
        day_dir = "log_data/" + day_str + "/"
        log_path = day_dir + day_str + "_" + time_str + "/"
        if not os.path.exists(day_dir):
            os.mkdir(day_dir)

        tb_writer = tf.summary.FileWriter(log_path)

        # Open text editor to write description of the run and commit it
        if '--temp' not in sys.argv:
            cmd = ['git', 'commit', __file__]
            os.environ['TF_LOG_DIR'] = log_path
            call(cmd)

        with tf.Session() as sess:

            tb_writer.add_graph(sess.graph)
            sess.run(self.init)
            c = 0

            for i in range(500):
                episode_iters = 0
                total_reward = 0
                observation = env.reset()
                # random initial action
                action = np.random.randint(0, self.action_dim)
                next_observation = env.step(action)[0]

                while True:

                    feed_dict = {
                        self.state: [observation],
                        self.true_next_state: [next_observation],
                        self.manual_action: action,
                        self.reward: total_reward
                    }

                    _, m_loss, next_state = sess.run([self.train_model, self.model_loss, self.predicted_next_state],
                                                     feed_dict)
                    # m_loss, next_state = sess.run([self.model_loss, self.predicted_next_state], feed_dict)

                    if self.on_policy_learning:
                        p_loss, action = sess.run([self.policy_loss, self.policy_action], feed_dict)

                    # comment this back in for off-policy learning
                    if not self.on_policy_learning:
                        e = np.random.rand()

                        # make random action X% of the time
                        if e > 0.4:
                            # sensible manual policy based only on velocity
                            if observation[1] < 0:
                                action = 0
                            else:
                                action = 2
                        else:
                            action = np.random.randint(0, self.action_dim)

                    if episode_iters % 10 == 0:
                        summary, step = sess.run([self.merged_summary, self.global_step], feed_dict)
                        tb_writer.add_summary(summary, step)

                    observation = next_observation
                    next_observation, reward, done, info = env.step(action)
                    total_reward += reward
                    episode_iters += 1

                    # if i % 10 == 0:
                    # env.render()

                    if done:
                        if episode_iters < 199:
                            c += 1
                        break

                if i % 50 == 0:
                    print("successes:", c, "iterations:", i)

        if upload:
            env.close()
            gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')


if __name__ == "__main__":
    hc = PolicyInModel()
    hc.main(upload=False)
