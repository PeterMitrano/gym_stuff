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
        self.transition_buffer_size = 1000
        self.transition_buffer = []
        self.stopping_criterion = 0.85  # no idea what this should be
        self.Q = {0: [0] * num_actions}  # start with one state
        self.root = Node(0)
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

    def best_action_idx(self, sense):
        node_id = self._sense_to_state_id(sense)
        action_idx = np.argmax(self.Q[node_id])
        return action_idx

    def _sense_to_state_leaf(self, sense):
        # traverse tree to find the appropriate state
        def __sense_to_state_leaf__(node, _sense):
            attribute = _sense[node.attribute_index]
            if attribute > node.attribute_value:
                if node.right is None:
                    return node
                else:
                    return __sense_to_state_leaf__(node.right, _sense)
            else:
                if node.left is None:
                    return node
                else:
                    return __sense_to_state_leaf__(node.left, _sense)
        return __sense_to_state_leaf__(self.root, sense)

    def _sense_to_state_id(self, sense):
        node = self._sense_to_state_leaf(sense)
        return node.id


    def _solve_mdp(self):
        """doesn't yet support stochastic environments, so state transition probabilities are not included here"""

        # not sure what to do here...
        pass

    def process(self):
        # here we want to compute new q values based on last computed V(s)
        # and we want to lookup all data points in a given state
        # data points is a dict of lists of tuples
        data_points = {}
        for trans in self.transition_buffer:
            s = self._sense_to_state_id(trans.I)
            s_prime = self._sense_to_state_id(trans.I_prime)
            Vs_prime = max(self.Q[s_prime])
            if s not in data_points:
                data_points[s] = []
            data_points[s].append((trans.I, trans.r + self.learning_rate * Vs_prime))


        # now for each leaf of data points look for slits in q(I,a)
        for data in data_points.values():
            max_diff = -sys.maxsize
            max_idx = -1
            splitting_dim = -1
            for dimension in range(self.sense_dimensions):
                # sort data of format by q(I,a)
                data_q = [datum[1] for datum in data]
                sorted_q = sorted(data_q)

                # loop over transitions and try splitting
                for idx in range(1, len(sorted_q)):
                    first = sorted_q[:idx]
                    last = sorted_q[idx:]
                    k_stat, p_value = stats.ks_2samp(first, last)

                    # .05 was used by Uther and Veloso
                    # k_stat must be between 0 and 1
                    if k_stat > max_diff and p_value < 0.05:
                        max_diff = k_stat
                        max_idx = idx
                        splitting_dim = dimension

            if max_diff > self.stopping_criterion:
                # introduce the split on the given dimension at the given q value
                splitting_sense = data[max_idx][0]
                node = self._sense_to_state_leaf(splitting_sense)
                node.attribute_index = splitting_dim
                node.attribute_value = splitting_sense[splitting_dim]

                # actually expand the tree
                node.left = Node(self.next_node_id)
                del(self.Q[node.id])
                self.Q[self.next_node_id] = [0] * self.num_actions
                self.next_node_id += 1
                node.right = Node(self.next_node_id)
                self.Q[self.next_node_id] = [0] * self.num_actions
                self.next_node_id += 1

        self._solve_mdp()


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
