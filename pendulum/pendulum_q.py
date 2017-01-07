import time
import numpy as np
import gym
from gym import wrappers
from q_learner import QLearner
import os

def main(upload=False):
    env = gym.make('Pendulum-v0')
    directory = '/tmp/' + os.path.basename(__file__) + '-' + str(int(time.time()))
    if upload:
        env = wrappers.Monitor(directory)(env)
        env.monitored = True
    else:
        env.monitored = False


    state_bounds = []
    state_bounds.append([-np.pi, np.pi, np.pi/24])
    state_bounds.append([-8, 8, 0.02])
    action_bounds = [-2, 2, 0.2]
    q_learner = QLearner(state_bounds, action_bounds)

    steps_between_render = 10000
    rewards = []
    max_trials = 100000
    print_step = 1000
    avg_reward = 0
    print('step, 100_episode_avg_reward')
    for i in range(max_trials):

        # decrease noise as we learn.
        noise_level = 36 / (12 + i)
        reward = q_learner.run_episode(env, noise_level, render=False)

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
    # bp.pack_ndarray_file(q_leaner.Q, 'q_table.bp')

    # upload/end monitoring
    if upload:
        env.close()
        gym.upload(directory, api_key='sk_8MyNtnorQEeNtKpCwk2S8g')


if __name__ == "__main__":
    main()
