import gym
import random
from continuous_utree import ContinuousUTree

def main():
    env = gym.make("CartPole-v0")

    sense_dims = env.observation_space.shape[0]
    utree = ContinuousUTree(sense_dimensions=sense_dims, num_actions=2)

    episode_iters = 200
    train_episodes = 10000
    process_iter = 10
    for i in range(train_episodes):

        j = 0
        observation = env.reset()
        episode_reward = 0
        while True:
            step = i * episode_iters + j

            # env.render()

            action = random.randint(0, 1)

            observation_prime, reward, done, info = env.step(action)
            episode_reward += reward

            utree.add_transition(observation, action, observation_prime, episode_reward)
            observation = observation_prime

            if done:
                utree.process()
                print("RANDOM POLICY SUCCEEED.", i, j)

            j += 1
            if j > episode_iters or done:
                break





if __name__ == "__main__":
    main()