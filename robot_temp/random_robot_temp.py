from time import sleep
import random
import gym

if __name__ == "__main__":
    env = gym.make("RobotTemperature-v0")

    for i in range(100):
        obs = env.reset()
        print(obs)
        env.render()
        input()
        j = 0
        while True:
            temp = random.randint(0, 2)
            move = int(random.random() + 0.5)
            obs, reward, done, info = env.step([move, temp])
            print(obs, reward)
            env.render()
            input()

            j += 1
            if j > 200:
                break

            if done:
                print("success!")
                break
