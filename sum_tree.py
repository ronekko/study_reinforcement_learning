# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 19:09:43 2021

@author: ryuhei
"""

import copy
import random
from typing import List

import hypothesis
import matplotlib.pyplot as plt
import numpy as np


class SumTreeQueue:
    def __init__(self, tree_depth: int, min_value: float = 0.0):
        self._tree_depth: int = tree_depth
        self._capacity: int = 2 ** tree_depth
        self._list_size: int = 2 * self._capacity
        self._tree: List[float] = [min_value] * self._list_size
        self._tree[0] = float('nan')
        self._leaf_index_tail: int = 0

    def __len__(self):
        return self._capacity

    def __getitem__(self, index):
        if not -self._capacity <= index < self._capacity:
            raise IndexError(
                f'index {index} is out of bounds with size {self._capacity}.')

        if index < 0:
            index += self._capacity

        return self._tree[self._capacity + index]

    def __setitem__(self, index, value):
        if not -self._capacity <= index < self._capacity:
            raise IndexError(
                f'index {index} is out of bounds with size {self._capacity}.')

        if index < 0:
            index += self._capacity

        leaf_index = self._capacity + index
        self._tree[leaf_index] = value
        node_idx = leaf_index // 2
        while node_idx >= 1:
            left = self._tree[node_idx * 2]
            right = self._tree[node_idx * 2 + 1]
            self._tree[node_idx] = left + right
            node_idx //= 2

    def __repr__(self):
        return 'SumTreeQueue' + repr(self._tree[self._capacity:])

    def push(self, value):
        self[self._leaf_index_tail] = value

        self._leaf_index_tail += 1
        if self._leaf_index_tail == self._capacity:
            self._leaf_index_tail = 0

    def copy_queue(self):
        """Create a deep-copy of the queue as a list."""
        return copy.deepcopy(self._tree[self._capacity:])

    def search(self, value):
        _tree = self._tree
        remainder = value
        idx = 1
        for _ in range(self._tree_depth):
            left = _tree[idx * 2]
            if remainder < left:
                idx = idx * 2  # left child
            else:
                remainder -= left
                idx = idx * 2 + 1  # right child
        leaf_idx = idx - self._capacity
        leaf_value = _tree[idx]
        return leaf_idx, leaf_value

    def get_total(self):
        return self._tree[1]

    def sample(self):
        u = random.uniform(0.0, self.get_total())
        return self.search(u)


if __name__ == '__main__':
    tree_depth = 7

    queue = SumTreeQueue(tree_depth)

    # Push random values to the queue
    capacity = len(queue)
    for i in range(capacity):
        priority = np.random.rand()
        queue.push(priority)

    indices = []
    for _ in range(10000):
        i, v = queue.sample()
        indices.append(i)

    priorities = queue.copy_queue()

    true_pmf = np.array(priorities) / np.sum(priorities)
    expected_histogram = len(indices) * true_pmf[:capacity]
    sample_histogram = np.bincount(indices, minlength=capacity)[:capacity]

    assert(
        hypothesis.chi_square_test(sample_histogram, expected_histogram, 0.01))

    bar_width = 0.35
    plt.bar(np.arange(capacity), expected_histogram, bar_width)
    plt.bar(np.arange(capacity) + bar_width, sample_histogram, bar_width)
    plt.legend(['expected', 'actual'])
    plt.show()
