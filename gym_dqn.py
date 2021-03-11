# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 16:52:17 2021

@author: ryuhei

TODO:
    - PrioritizedReplayBufferのTD誤差を更新せよ
"""

import itertools
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import gym  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        fc1_dim = 32
        fc2_dim = 64
        fc3_dim = 128
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc_out = nn.Linear(fc3_dim, action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        return self.fc_out(x)


@dataclass
class Transition:
    s: np.array
    a: int
    r: float
    s_next: np.array
    terminal: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: List[Transition] = []
        self._capacity: int = capacity
        self._tail_index: int = 0

    def push(self, transition: Transition):
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        self._buffer[self._tail_index] = transition
        self._tail_index += 1
        if self._tail_index == self._capacity:
            self._tail_index = 0

    def sample_batch(self, batch_size: int):
        buffer_size = len(self._buffer)
        if buffer_size < batch_size:
            raise ValueError(f'batch size {batch_size} must not be larger than'
                             f' buffer size {buffer_size}.')
        indices = np.random.choice(buffer_size, batch_size, replace=False)
        return self._make_batch(indices)

    def _make_batch(self, indices: List[int]):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        terminal_list = []
        for i in indices:
            transition = self._buffer[i]
            s_list.append(transition.s)
            a_list.append([transition.a])
            reward = transition.r
            # if transition.terminal:
            #     if (-2.4 < transition.s_next[0] < 2.4 and
            #             -0.2 < transition.s_next[2] < 0.2):
            #         reward = 1
            #     else:
            #         reward = -1
            # else:
            #     reward = 0
            r_list.append([reward])
            s_next_list.append(transition.s_next)
            terminal_list.append(transition.terminal)
        s = torch.tensor(s_list)
        a = torch.tensor(a_list)
        r = torch.tensor(r_list)
        s_next = torch.tensor(s_next_list)
        terminal = torch.tensor(terminal_list)
        return s, a, r, s_next, terminal

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return str(self._buffer)


if __name__ == '__main__':
    # CartPose-v1 (https://github.com/openai/gym/wiki/CartPole-v0)
    # Episode Termination condition:
    # 1. Pole Angle is more than ±12°
    # 2. Cart Position is more than ±2.4 (center of the cart reaches the edge
    #    of the display)
    # 3. Episode length is greater than 200 (500 for v1).
    env_name = 'CartPole-v0'
    env = gym.make(env_name)  # state(pos, vel, angle, angular vel)

    # Hyperparameters
    gamma = 0.999  # Rate of reward discount
    num_learning_steps = 400000
    replay_buffer_size = 2**14  # 2**14 = 16384
    initial_exploration_steps = 500
    q_update_interval = 1
    q_target_update_interval = 2000

    eps_init = 1.0
    eps_decay_rate = 0.98
    eps_decay_start = initial_exploration_steps
    eps_decay_period = 2
    eps_bottom = 0.1
    eps_high = 0.4
    eps_high_ratio = 0.05  # 経験に悪例が無くなってしまわないように定期的にepsの高い期間を設ける

    # Network training hyper params
    lr = 1e-4  # Learning rate for updating Q function
    batch_size = 128

    # Network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q = Net(state_dim, action_dim)
    optimizer = torch.optim.RMSprop(q.parameters(), lr)

    q_target = Net(state_dim, action_dim).requires_grad_(False)
    q_target.load_state_dict(q.state_dict())
    q_target.eval()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # Training
    try:
        total_steps = 0
        eps = eps_init
        returns = []
        epsilons = []
        episode_steps = []
        step_losses = []
        episode_losses = []
        transition_wise_errors = []
        for episode in itertools.count():

            if total_steps > eps_decay_start:
                if episode % eps_decay_period == 0:
                    eps_updates = episode // eps_decay_period
                    eps = eps_init * (eps_decay_rate ** eps_updates)
                if eps < eps_bottom:
                    eps = eps_bottom
                # # 経験に悪例が常に一定割合存在するように定期的にepsが高い期間を設ける
                # if (total_steps / replay_buffer_size) % 1 < eps_high_ratio:
                #     eps = eps_high

            initial_obs = env.reset()
            transitions = []
            losses_in_episode = []
            s_current = initial_obs.astype(np.float32)
            total_reward = 0
            for step in itertools.count():
                epsilons.append(eps)
                # Choose an action from {0, 1}, where
                # 0 or 1 are left or right acceleration respectively.
                # a = env.action_space.sample()
                if random.random() < eps:
                    a = random.randint(0, 1)
                else:
                    with torch.no_grad():
                        action_values = q(torch.tensor([s_current]))
                        a = action_values.argmax(1).item()

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

                s_next = obs.astype(np.float32)
                trans = Transition(s_current, a, reward, s_next, done)
                replay_buffer.push(trans)

                s_current = s_next
                total_reward += reward
                total_steps += 1
                transitions.append(trans)

                if done:
                    break

                # Only explore for the initial exploration phase.
                if (total_steps < initial_exploration_steps or
                        total_steps % q_update_interval != 0):
                    continue

                ####################
                # Update Q function

                # Update target Q function
                if total_steps % q_target_update_interval == 0:
                    q_target.load_state_dict(q.state_dict())

                batch = replay_buffer.sample_batch(batch_size)
                s, a, r, s_next, terminal = batch

                # compute target value
                with torch.no_grad():
                    action_values_next = q_target(s_next)
                    max_action_values_next = action_values_next.amax(1)
                    max_action_values_next[terminal] = 0.0
                    t = reward + gamma * max_action_values_next
                action_values = q(s)
                y = action_values.gather(1, a).flatten()
                loss = F.smooth_l1_loss(y, t)
                # loss = F.mse_loss(y, t)
                errors = np.abs(y.detach().numpy() - t.detach().numpy())
                transition_wise_errors.append(errors)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_ = loss.item()
                step_losses.append(loss_)
                losses_in_episode.append(loss_)
                # print(loss_)

            returns.append(total_reward)
            episode_steps.append(total_steps)
            episode_losses.append(np.mean(losses_in_episode))
            if episode % 10 == 0:
                plt.plot(step_losses, '.', markersize=1)
                plt.show()
                plt.close()

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
    plt.plot(np.array(episode_losses) * 200, '.', markersize=1)
    plt.legend(['returns', 'losses'])
    plt.grid()
    plt.show()

    # Evaluation
    try:
        for test_episode in itertools.count():
            initial_obs = env.reset()
            s = initial_obs.astype(np.float32)
            total_reward = 0
            images = []
            for test_step in itertools.count():
                env.render()
                images.append(env.render('rgb_array'))
                with torch.no_grad():
                    action_values = q(torch.tensor([s]))
                    a = action_values.argmax(1).item()
                outcome = env.step(a)
                obs, reward, done, info = outcome
                s = obs.astype(np.float32)
                total_reward += reward
                if done:
                    break

            # print(transitions)
            print(f'# {test_episode}: R = {total_reward}, eps = 0')

            # Save the images of this episode.
            current_datetime = time.strftime("%Y%m%d-%H%M%S")
            episode_dirpath = Path(env_name, current_datetime)
            episode_dirpath.mkdir(parents=True, exist_ok=True)
            print(f'Saving images to {episode_dirpath.absolute()} ...', end='')
            n_digits = int(np.ceil(np.log10(len(images) + 1)))
            for filename, image in enumerate(images):
                filepath = episode_dirpath / f'{filename:0>{n_digits}}.png'
                plt.imsave(filepath, image)
            print('done.')
            print()

    except KeyboardInterrupt:
        print('Ctrl+c.')

    env.close()
