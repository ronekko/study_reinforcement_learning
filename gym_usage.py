# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:52:17 2021

@author: ryuhei
"""

import gym

if __name__ == '__main__':
    # Gallery of environments: https://gym.openai.com/envs/#classic_control

    env = gym.make('CartPole-v1')  # state(pos, vel, angle, angular vel)
    # env = gym.make('Pendulum-v0')
    # env = gym.make('MsPacman-v0')
    # env = gym.make('Pong-v0')
    env.reset()

    for _ in range(10):
        env.reset()
        states = []
        rewards = []
        for _ in range(200):
            env.render()

            # Choose an action from {0, 1}, where
            # 0 or 1 are left or right acceleration respectively.
            action = env.action_space.sample()

            # Do the action and receive an outcome.
            # An outcome consists of
            #   - observation (float np.array of length 4):
            #       [cart pos, cart vel, pole angle, pole angular vel]
            #   - reward (float): constant 1.
            #   - done (bool): True if the pole angle has exceeded the limits.
            #   - info (dict): Additional information.
            outcome = env.step(action)
            obs, reward, done, info = outcome

            states.append(obs)
            rewards.append(reward)

            if done:
                break

        env.close()
