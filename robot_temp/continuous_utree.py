""" Based off the paper by Uther and Veloso, CMU, 1998 """

import matplotlib.pyplot as plt
from scipy import stats
from collections import namedtuple
import numpy as np
import sys


Transition = namedtuple("Transition", ['I', 'a', 'I_prime', 'r'])


class Node:

    def __init__(self, node_id, attr_index=0, attr_val=0):
        self.attribute_index = attr_index
        self.attribute_value = attr_val
        self.left = None
        self.right = None
        self.id = node_id
        self.rewards = []
        self.avg_reward = 0

        if not hasattr(Node, 'num_actions'):
            raise RuntimeError("You must call Node.set_num_actions first.")

        self.q = np.zeros(Node.num_actions)
        self.visits = np.zeros(Node.num_actions)

    @staticmethod
    def set_num_actions(num_actions):
        Node.num_actions = num_actions

    def __repr__(self, level=0):
        ret = "\t" * level + "%i, %3.3f" % (self.attribute_index, self.attribute_value) + "\n"
        if self.left is not None and self.right is not None:
            ret += self.left.__repr__(level + 1)
            ret += self.right.__repr__(level + 1)
        return ret


class ContinuousUTree:

    def __init__(self, sense_dimensions, num_actions):
        self.sense_dimensions = sense_dimensions
        self.num_actions = num_actions
        self.transition_buffer_size = 2000
        self.transition_buffer = []
        self.stopping_criterion = 0.85  # no idea what this should be
        Node.set_num_actions(num_actions)
        self.root = Node(0)
        self.leaf_states = [self.root]
        self.next_node_id = 1
        self.learning_rate = 0.9

    def add_transition(self, sense, action, sense_prime, reward):
        """ action should be an index, not the raw action value. """
        trans = Transition(sense, action, sense_prime, reward)
        if len(self.transition_buffer) < self.transition_buffer_size:
            self.transition_buffer.append(trans)
        else:
            # replace random transition
            idx = np.random.randint(0, self.transition_buffer_size)
            self.transition_buffer[idx] = trans

        state = self._sense_to_state(sense)
        next_state = self._sense_to_state(sense_prime)

        for action in range(self.num_actions):
            alpha = 1 / (1 + state.visits[action])
            expected_future_reward = np.max(next_state.q)
            new_q = reward + self.learning_rate * expected_future_reward
            state.q[action] = (1 - alpha) * state.q[action] + alpha * new_q
            state.visits[action] += 1

    def best_action_idx(self, sense):
        state = self._sense_to_state(sense)
        return np.argmax(state.q)

    def _sense_to_state(self, sense):
        # traverse tree to find the appropriate state
        def __sense_to_state_state__(node, _sense):
            attribute = _sense[node.attribute_index]
            if attribute >= node.attribute_value:
                if node.right is None:
                    return node
                else:
                    return __sense_to_state_state__(node.right, _sense)
            else:
                if node.left is None:
                    return node
                else:
                    return __sense_to_state_state__(node.left, _sense)
        return __sense_to_state_state__(self.root, sense)

    def _sense_to_state_id(self, sense):
        node = self._sense_to_state(sense)
        return node.id

    def _update_q_values(self):
        for state in self.leaf_states:
            for action in self.num_actions:
                alpha = 1 / 1 + (state.visits[action])
                expected_future_reward = np.max(next_state.q)
                state.avg_reward = np.average(state.rewards)
                new_q = state.avg_reward + self.learning_rate * expected_future_reward
                state.q[action] = (1 - alpha) * self.q[action] + alpha * new_q

                # could also do this in every transition?
                state.visits[action] += 1

    def process(self):
        # here we want to compute new q values based on last computed V(s)
        # and we want to lookup all data points in a given state
        # data points is a dict of lists of tuples
        data_points = {}
        for trans in self.transition_buffer:
            s_state = self._sense_to_state(trans.I)
            s_prime_state = self._sense_to_state(trans.I_prime)
            expected_future_reward = np.max(s_prime_state.q)
            if s_state.id not in data_points:
                data_points[s_state.id] = []
            data_points[s_state.id].append((trans.I, trans.r + self.learning_rate * expected_future_reward))

        # now for each state of data points look for slits in q(I,a)
        did_split = False
        for data in data_points.values():
            max_diff = -sys.maxsize
            max_idx = -1
            splitting_dim = -1
            for dimension in range(self.sense_dimensions):
                # sort data of format by q(I,a)
                sorted_data = sorted(data, key=lambda datum: datum[1])
                sorted_r = [datum[1] for datum in sorted_data]

                # loop over transitions and try splitting
                for idx in range(1, len(sorted_r)):
                    first = sorted_r[:idx]
                    last = sorted_r[idx:]
                    k_stat, p_value = stats.ks_2samp(first, last)

                    # TODO: visualize first/last to see if things are working

                    # .05 was used by Uther and Veloso
                    # k_stat must be between 0 and 1
                    if k_stat > max_diff and p_value < 0.05:
                        max_diff = k_stat
                        max_idx = idx
                        max_first = first
                        max_last = last
                        splitting_dim = dimension

            if max_diff > self.stopping_criterion:
                did_split = True
                # introduce the split on the given dimension at the given q value
                splitting_sense = sorted_data[max_idx][0]
                state = self._sense_to_state(splitting_sense)
                state.attribute_index = splitting_dim
                state.attribute_value = splitting_sense[splitting_dim]

                # actually expand the tree
                state_idx = state.id
                self.leaf_states.remove(state)

                # add left child
                self.next_node_id += 1
                state.left = Node(self.next_node_id)
                self.leaf_states.append(state.left)

                # add right child
                self.next_node_id += 1
                state.right = Node(self.next_node_id)
                self.leaf_states.append(state.right)

        # TODO: figure out what we're supposed to do here...
        # self._update_q_values()

        return did_split


def main():
    # 2d sensory input, and actions are 0, 1, or 2
    tree = ContinuousUTree(2, 3)
    min_action = 0
    action_step = 1

    sense = (0, 0)

    for _ in range(5):
        # make up some 2d data + reward
        for x in np.arange(-2, 2, 0.1):
            for x_dot in np.arange(-2, 2, 0.1):
                if x > 0 and x_dot > 0:
                    action = 0
                elif x < 0 and x_dot < 0:
                    action = 2
                else:
                    action = 1
                sense_prime = (x, x_dot)
                reward = action + x - x_dot
                action_idx = (action - min_action) // action_step
                tree.add_transition(sense, action_idx, sense_prime, reward)

                sense = sense_prime

        tree.process()


if __name__ == "__main__":
    main()
