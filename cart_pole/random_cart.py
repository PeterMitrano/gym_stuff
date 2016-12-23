#!/usr/bin/python3
import gym
env = gym.make('CartPole-v0')
env.reset()

print "angle, reward"
for _ in range(1000):
    env.render()
    observations, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print observations[2], "," , reward

    if done:
        print "done!"
        break
