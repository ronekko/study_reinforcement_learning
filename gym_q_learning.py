# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:52:17 2021

@author: ryuhei
"""

import itertools
import random
from dataclasses import dataclass
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Transition:
    s: List[int]
    a: int
    s_next: List[int]
    r: float


def obs_to_state(obs, num_bins: int):
    max_pos = 2.4
    min_pos = -max_pos
    max_vel = 3.0
    min_vel = -max_vel
    max_angle = np.deg2rad(12.0)
    min_angle = -max_angle
    max_anguvel = 2.4
    min_anguvel = -max_anguvel
    bins = np.linspace(-0.95, 0.95, num_bins - 1)

    pos, vel, angle, anguvel = obs

    pos_bin = sum(bins < pos / ((max_pos - min_pos) / 2))
    vel_bin = sum(bins < vel / ((max_vel - min_vel) / 2))
    angle_bin = sum(bins < angle / ((max_angle - min_angle) / 2))
    anguvel_bin = sum(bins < anguvel / ((max_anguvel - min_anguvel) / 2))
    return pos_bin, vel_bin, angle_bin, anguvel_bin


if __name__ == '__main__':
    # CartPose-v1 (https://github.com/openai/gym/wiki/CartPole-v0)
    # Episode Termination condition:
    # 1. Pole Angle is more than ±12°
    # 2. Cart Position is more than ±2.4 (center of the cart reaches the edge
    #    of the display)
    # 3. Episode length is greater than 200 (500 for v1).
    env = gym.make('CartPole-v1')  # state(pos, vel, angle, angular vel)
    # env.unwrapped.theta_threshold_radians = np.rad2deg(24.0)
    env.reset()

    num_bins = 11  # Number of state quantization
    gamma = 0.99  # Rate of reward discount
    lr = 0.1  # Learning rate for updating Q function
    eps_init = 1.0
    eps_linear_decay = False
    num_learning_steps = 2000000

    if eps_linear_decay:
        eps_decay_rate = 0.98
        eps_decay_start = num_learning_steps // 30000
        eps_decay_period = 50000
    else:
        num_learning_episodes = 10000
        eps_decay_rate = 0.98
        eps_decay_start = 0
        eps_decay_period = 50

    num_actions = 2
    q = np.full((num_bins, num_bins, num_bins, num_bins, num_actions), 10.0,
                np.float64).tolist()
    # q = np.random.rand(num_bins, num_bins, num_bins, num_bins, num_actions
    #                    ).astype(np.float64).tolist()

    try:
        total_steps = 0
        eps = eps_init
        returns = []
        epsilons = []
        episode_steps = []
        for episode in itertools.count():
            if eps_linear_decay:
                if total_steps > eps_decay_start:
                    decay_span = (num_learning_steps - eps_decay_start)
                    decay_rate = (total_steps - eps_decay_start) / decay_span
                    eps = eps_init * (1.0 - decay_rate) ** 0.5
            else:
                if episode > eps_decay_start:
                    if episode % eps_decay_period == 0:
                        eps *= eps_decay_rate

            initial_obs = env.reset()
            s0 = obs_to_state(initial_obs, num_bins)
            transitions = []
            s = s0
            total_reward = 0
            for step in itertools.count():
                epsilons.append(eps)
                # Choose an action from {0, 1}, where
                # 0 or 1 are left or right acceleration respectively.
                # a = env.action_space.sample()
                if random.random() < eps:
                    a = random.randint(0, 1)
                else:
                    policy = q[s[0]][s[1]][s[2]][s[3]]
                    a = 0 if policy[0] > policy[1] else 1

                # Do the action and receive an outcome.
                # An outcome consists of
                #   - observation (float np.array of length 4):
                #       [cart pos, cart vel, pole angle, pole angular vel]
                #       - angle: clockwise is positive.
                #   - reward (float): constant 1.
                #   - done (bool): If the pole angle has exceeded the limits.
                #   - info (dict): Additional information.
                outcome = env.step(a)
                obs, reward, done, info = outcome

                s_next = obs_to_state(obs, num_bins)
                trans = Transition(s, a, s_next, reward)
                transitions.append(trans)

                # Update Q(s, a)
                q_current = q[s[0]][s[1]][s[2]][s[3]][a]
                q_next_max = max(q[s_next[0]][s_next[1]][s_next[2]][s_next[3]])
                q[s[0]][s[1]][s[2]][s[3]][a] += lr * (
                    reward + gamma * q_next_max - q_current)

                s = s_next
                total_reward += reward
                total_steps += 1

                if done:
                    break
            returns.append(total_reward)
            episode_steps.append(total_steps)

            print(transitions)
            print(f'# {episode}: R = {total_reward}, eps = {eps}'
                  f', steps = {total_steps}')
            print()

            if total_steps > num_learning_steps:
                break

    except KeyboardInterrupt:
        print('Ctrl+c.')

    env.close()

    caption = (f'{lr=}, eps_decay(rate={eps_decay_rate},'
               f' period={eps_decay_period})')
    plt.title(caption)
    plt.plot(returns, '.', markersize=1)
    plt.show()

    try:
        while True:
            initial_obs = env.reset()
            s0 = obs_to_state(initial_obs, num_bins)
            s = s0
            total_reward = 0
            transitions = []
            while True:
                env.render()
                policy = q[s[0]][s[1]][s[2]][s[3]]
                a = 0 if policy[0] > policy[1] else 1
                outcome = env.step(a)
                obs, reward, done, info = outcome
                s_next = obs_to_state(obs, num_bins)
                trans = Transition(s, a, s_next, reward)
                transitions.append(trans)
                total_reward += reward

                s = s_next
                if done:
                    break

            print(transitions)
            print(f'# {episode}: R = {total_reward}, eps = 0')
            print()

    except KeyboardInterrupt:
        print('Ctrl+c.')

    env.close()
