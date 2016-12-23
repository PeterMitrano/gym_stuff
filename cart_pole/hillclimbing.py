#!/usr/bin/python3
import matplotlib.pyplot as plt
import gym
import numpy as np


class HillClimbing:

    @staticmethod
    def run_episode(env, parameters):
        observation = env.reset()
        total_reward = 0
        for _ in range(200):
            env.render()
            # compute linear combination of parameters and observations
            if np.matmul(parameters, observation) < 0:
                action = 0
            else:
                action = 1
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    @staticmethod
    def normalize(x):
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        else:
            return x/norm

    @staticmethod
    def train():
        env = gym.make('CartPole-v0')

        # parameters start as random values between -1 and 1
        # EX of good params: [-0.1906  0.2306  0.4481  0.2017]
        observation_n = env.observation_space.shape[0]
        parameters = np.random.rand(observation_n) * 2 - 1
        best_reward = 0

        # plotting stuff
        steps_between_plot = 40
        plt.ion()

        # 2000 episodes
        rewards = []
        max_trials = 1000
        for i in range(max_trials):
            noise_scaling = 1 - (i / max_trials)
            new_params = parameters + (np.random.rand(observation_n) * 2 - 1) * noise_scaling
            # normalize it because it shouldn't effect things
            new_params = HillClimbing.normalize(new_params)
            print(new_params)
            reward = HillClimbing.run_episode(env, new_params)
            rewards.append(reward)

            if i % steps_between_plot == 0:
                plt.plot(rewards)
                plt.pause(0.05)
            print("[%i] episode reward %d, best so far %d" % (i, reward, best_reward))
            if reward > best_reward:
                best_reward = reward
                parameters = new_params
                if reward == 200:
                    print("game has been solved!")
                    break

        np.savetxt('rewards.csv', rewards, delimiter=',')
        input("press enter to exit")
        return parameters, best_reward


if __name__ == "__main__":
    hc = HillClimbing()
    r = hc.train()
    print(r)
