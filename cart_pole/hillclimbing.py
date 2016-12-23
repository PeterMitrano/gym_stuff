#!/usr/bin/python3
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
import gym
import numpy as np

class HillClimbing:

    def run_episode(self, env, parameters):
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

    def train(self):
        env = gym.make('CartPole-v0')

        # parameters are 4 random values between -1 and 1
        parameters = np.random.rand(4) * 2 - 1
        best_reward = 0

        # plotting stuff
        steps_between_plot = 40
        plt.ion()

        # 2000 episodes
        rewards = []
        for i in range(2000):
            noise_scaling = 0.05
            new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
            reward = self.run_episode(env, new_params)
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

        plt.savefig('fig.png')
        input("press enter to exit")
        return best_reward


if __name__ == "__main__":
    hc = HillClimbing()
    r = hc.train()
    print(r)
