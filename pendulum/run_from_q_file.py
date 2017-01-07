import gym
from time import sleep
from pendulum_q import QLearner
import bloscpack as bp

if __name__ == "__main__":
    filename = 'q_table.bp'
    q_table = bp.unpack_ndarray_file(filename)

    env = gym.make("Pendulum-v0")
    ql = QLearner()
    ql.init_q_table(q_table)

    while True:
        for i in range(10):
            obs = env.reset()
            total_reward = 0
            for j in range(500):
                env.render()
                _, action, _ = ql.q_policy(obs, noise_level=0)
                obs, reward, done, info = env.step([action])
                # sleep(0.1)
                total_reward += reward
            print(total_reward)

        input("press enter for 10 more runs")
