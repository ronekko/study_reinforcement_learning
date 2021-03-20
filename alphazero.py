# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:33:42 2021

@author: ryuhei
"""

import copy
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
        self.fc_1 = nn.Linear(state_dim, fc1_dim)
        self.fc_2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc_3 = nn.Linear(fc2_dim, fc3_dim)
        self.fc_policy_1 = nn.Linear(fc3_dim, fc3_dim)
        self.fc_policy_out = nn.Linear(fc3_dim, action_dim)
        self.fc_value_1 = nn.Linear(fc3_dim, fc3_dim)
        self.fc_value_out = nn.Linear(fc3_dim, 1)

    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)

        x = self.fc_2(x)
        x = F.relu(x)

        x = self.fc_3(x)
        x = F.relu(x)

        p = self.fc_policy_1(x)
        p = F.relu(p)
        p = self.fc_policy_out(p)
        p = F.softmax(p, dim=1)

        v = self.fc_value_1(x)
        v = F.relu(v)
        v = self.fc_value_out(v)
        v = torch.sigmoid(v)
        return p, v


@dataclass
class DataPoint:
    s: np.array
    pi: np.array
    z: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer: List[DataPoint] = []
        self._capacity: int = capacity
        self._tail_index: int = 0

    def push(self, data_point: DataPoint):
        if len(self._buffer) < self._capacity:
            self._buffer.append(data_point)
        self._buffer[self._tail_index] = data_point
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
        pi_list = []
        z_list = []
        for i in indices:
            data_point = self._buffer[i]
            s_list.append(data_point.s)
            pi_list.append(data_point.pi)
            z_list.append([data_point.z])
        s = torch.tensor(s_list)
        pi = torch.tensor(pi_list)
        z = torch.tensor(z_list)
        return s, pi, z

    def __len__(self):
        return len(self._buffer)

    def __repr__(self):
        return str(self._buffer)


class AlphZeroTree:
    def __init__(self, simulator, predictor, s, dim_action, c_puct,
                 n_simulations, temperature, root=None):
        self.simulator = simulator
        self.predictor = predictor
        self.dim_action = dim_action
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.temperature = temperature

        if root is None:
            with torch.no_grad():
                p, v = self.predictor(torch.tensor([s]))
            p = p[0].numpy()
            v = v.item()
            self.root = Node(s, p, v, False, dim_action, None)
        else:
            root.parent = None
            self.root = root

    def search(self):
        for i in range(self.n_simulations):
            # print(f'{i}: ', end='')
            node = self.root
            total_reward = 0
            for depth in itertools.count():
                # print(f'{depth} ', end='')
                a = self.select(node)
                outcome = self.simulator.move(node.s, a)
                obs, r, done, info = outcome
                s_next = obs.astype(np.float32)
                total_reward += r
                child_node = node.children[a]
                if child_node is not None:  # non-leaf node
                    node = child_node
                    if node.terminal:
                        node.backup()
                        break
                elif done:  # leaf node and terminal state
                    z = total_reward / self.simulator.max_return
                    new_node = Node(
                        s_next, None, z, True, self.dim_action, node)
                    node.children[a] = new_node
                    new_node.backup()
                    break
                else:  # leaf node and non-terminal state
                    with torch.no_grad():
                        p, v = self.predictor(torch.tensor([s_next]))
                    p = p[0].numpy()
                    v = v.item()
                    new_node = Node(
                        s_next, p, v, False, self.dim_action, node)
                    node.children[a] = new_node
                    new_node.backup()
                    break
            # print()

    def select(self, node):
        N = []
        W = []
        for child in node.children:
            if child:
                N.append(child.N)
                W.append(child.W)
            else:
                N.append(0.0)
                W.append(0.0)
        N = np.array(N)
        W = np.array(W)

        # Initial selection from this node is uniform random choice,
        # because both Q and U are zero (i.e., all actions are tie).
        if N.sum() == 0:
            return np.random.choice(self.dim_action)
        # print(node.p)
        U = self.c_puct * node.p * (N.sum() ** 0.5) / (N + 1)
        Q = np.where(N > 0, W / N, 0)
        a = np.argmax(Q + U)
        return a

    def compute_pi(self):
        Ns = []
        for child in self.root.children:
            if child:
                Ns.append(child.N)
            else:
                Ns.append(0)
        Ns = np.array(Ns) + 1
        prob = Ns ** (1.0 / self.temperature)
        prob /= prob.sum()
        return prob


class Node:
    def __init__(self, s, p, v, terminal, dim_action, parent):
        self.s = s
        self.p = p
        self.v = v
        self.terminal = terminal
        self.N = 0.0
        self.W = 0.0
        self.children = [None] * dim_action
        self.parent = parent

    def backup(self):
        node = self
        while node is not None:
            node.N += 1.0
            node.W += self.v
            node = node.parent

    def __repr__(self):
        return (f'Node(s={self.s}, p={self.p}, v={self.v}, N={self.N}, '
                f'W={self.W}, terminal={self.terminal})')


class GymSimulator:
    def __init__(self, env, max_return):
        self.env = copy.deepcopy(env)
        self.max_return = max_return

    def move(self, s, a):
        self.env.reset()
        self.env.env.state = s
        return self.env.step(a)


def cross_entropy_loss(p_pred, p_target):
    return -torch.sum(p_target * torch.log(p_pred), dim=1).mean()


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
    num_learning_steps = 400000
    replay_buffer_size = 2**14  # 2**14 = 16384
    initial_exploration_steps = 500
    q_update_interval = 1

    c_puct = 1.0
    n_simulations = 100
    temperature = 1

    # Network training hyper params
    lr = 1e-2  # Learning rate for updating Q function
    batch_size = 128

    # Network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    predictor = Net(state_dim, action_dim)
    optimizer = torch.optim.Adam(predictor.parameters(), lr)

    # q_target = Net(state_dim, action_dim).requires_grad_(False)
    # q_target.load_state_dict(q.state_dict())
    # q_target.eval()
    replay_buffer = ReplayBuffer(replay_buffer_size)

    if env_name == 'CartPole-v0':
        max_return = 200
    elif env_name == 'CartPole-v1':
        max_return = 500
    else:
        raise ValueError(f'Environment "{env_name}" is not supported.')

    simulator = GymSimulator(env, max_return)

    # Training
    try:
        total_steps = 0
        returns = []
        episode_steps = []
        losses = []
        for episode in itertools.count():
            print(f'# {episode}: ', end='')
            initial_obs = env.reset()
            states = []
            pis = []
            next_root = None
            s_current = initial_obs.astype(np.float32)
            total_reward = 0
            for step in itertools.count():
                env.render()
                print(f'{step} ', end='')

                tree = AlphZeroTree(
                    simulator, predictor, s_current, action_dim, c_puct,
                    n_simulations, temperature, next_root)
                tree.search()
                pi = tree.compute_pi()
                a = np.random.choice(action_dim, p=pi)

                # Reuse the subtree of selected action
                next_root = tree.root.children[a]
                # Even if the root has children that have not been expanded,
                # all actions still have nonzero probability and can be drawn.
                if next_root and next_root.terminal:
                    next_root = None

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

                states.append(s_current)
                pis.append(pi)

                s_current = s_next
                total_reward += reward
                total_steps += 1

                if done:
                    break

            returns.append(total_reward)
            episode_steps.append(total_steps)
            z = total_reward / max_return  # normalized return
            for s, pi in zip(states, pis):
                data = DataPoint(s, pi, z)
                replay_buffer.push(data)

            print(f': R = {total_reward}, steps = {total_steps}')
            print()

            # Only explore for the initial exploration phase.
            if total_steps < initial_exploration_steps:
                continue

            ####################
            # Update predictor
            for k in range(100):
                batch = replay_buffer.sample_batch(batch_size)
                s, p_target, v_target = batch

                # compute target value
                p_pred, v_pred = predictor(s)
                loss_v = F.mse_loss(v_pred, v_target)
                loss_p = cross_entropy_loss(p_pred, p_target)
                loss = loss_v + loss_p

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_ = loss.item()
                losses.append(loss)

            if episode % 10 == 0:
                caption = (f'{lr=}')
                fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
                # axes[0].set_title(caption)
                window_size = 30
                ax0.plot(returns, '.', markersize=2)
                ax0.plot(np.convolve(
                    returns, np.ones(window_size), 'valid')/window_size, '-')
                ax0.legend(['returns', f'avg (window={window_size})'])
                ax0.grid()
                # axes[1].set_title(caption)
                ax1.plot(losses, '.', markersize=1)
                ax1.legend(['losses'])
                ax1.grid()
                plt.tight_layout()
                plt.show()
                plt.close()

            if total_steps > num_learning_steps:
                break

    except KeyboardInterrupt:
        print('Ctrl+c.')

    env.close()

    caption = (f'{lr=}')
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0].set_title(caption)
    window_size = 30
    ax0.plot(returns, '.', markersize=2)
    ax0.plot(np.convolve(
        returns, np.ones(window_size), 'valid')/window_size, '-')
    ax0.legend(['returns', f'avg (window={window_size})'])
    ax0.grid()
    # axes[1].set_title(caption)
    ax1.plot(losses, '.', markersize=1)
    ax1.legend(['losses'])
    ax1.grid()
    plt.tight_layout()
    plt.show()
    plt.close()

    # # Evaluation
    # try:
    #     for test_episode in itertools.count():
    #         initial_obs = env.reset()
    #         s = initial_obs.astype(np.float32)
    #         total_reward = 0
    #         images = []
    #         for test_step in itertools.count():
    #             env.render()
    #             images.append(env.render('rgb_array'))
    #             with torch.no_grad():
    #                 action_values = q(torch.tensor([s]))
    #                 a = action_values.argmax(1).item()
    #             outcome = env.step(a)
    #             obs, reward, done, info = outcome
    #             s = obs.astype(np.float32)
    #             total_reward += reward
    #             if done:
    #                 break

    #         # print(transitions)
    #         print(f'# {test_episode}: R = {total_reward}, eps = 0')

    #         # Save the images of this episode.
    #         current_datetime = time.strftime("%Y%m%d-%H%M%S")
    #         episode_dirpath = Path(env_name, current_datetime)
    #         episode_dirpath.mkdir(parents=True, exist_ok=True)
    #         print(f'Saving images to {episode_dirpath.absolute()} ...', end='')
    #         n_digits = int(np.ceil(np.log10(len(images) + 1)))
    #         for filename, image in enumerate(images):
    #             filepath = episode_dirpath / f'{filename:0>{n_digits}}.png'
    #             plt.imsave(filepath, image)
    #         print('done.')
    #         print()

    # except KeyboardInterrupt:
    #     print('Ctrl+c.')

    # env.close()
