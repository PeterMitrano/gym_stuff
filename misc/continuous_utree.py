# Based off the paper by Uther and Veloso, CMU, 1998

from scipy import stats
from collections import namedtuple
import numpy as np
import sys


Transition = namedtuple("Transition", ['I', 'a', 'I_prime', 'r'])


class Node:

    def __init__(self, attr_index=0, attr_val=0, node_id=0):
        self.attribute_index = attr_index
        self.attribute_value = attr_val
        self.left = None
        self.right = None
        self.id = node_id


class ContinuousUTree:

    def __init__(self, sense_dimensions, num_actions):
        self.sense_dimensions = sense_dimensions
        self.num_actions = num_actions
        self.transition_buffer_size = 1000
        self.transition_buffer = []
        self.stopping_criterion = 1.0  # no idea what this should be
        self.Q = [[0] * num_actions]
        self.root = Node(node_id=0)
        self.next_node_id = 1
        self.learning_rate = 0.9

    def add_transition(self, sense, action, sense_prime, reward):
        trans = Transition(sense, action, sense_prime, reward)
        if len(self.transition_buffer) < self.transition_buffer_size:
            self.transition_buffer.append(trans)
        else:
            # replace random transition
            idx = np.random.randint(0, self.transition_buffer_size)
            self.transition_buffer[idx] = trans

    def sense_to_state_id(self, sense):
        # traverse tree to find the appropriate state
        def _sense_to_state_id(node, _sense):

            attribute = _sense[node.attribute_index]
            if attribute > node.attribute_value:
                if not node.right:
                    return node.id
                else:
                    return _sense_to_state_id(node.right, _sense)
            else:
                if not node.left:
                    return node.id
                else:
                    return _sense_to_state_id(node.left, _sense)

        return _sense_to_state_id(self.root, sense)

    def compute_qvalues(self, transitions):
        # here we want to compute new q values based on last computed V(s)
        # ie. (I, a, I', r) -> (I, a, q(I, a))
        for trans in transitions:
            s = self.sense_to_state_id(trans.I)
            s_prime = self.sense_to_state_id(trans.I_prime)
            Vs_prime = max(self.Q[s_prime])
            self.Q[s][trans.a] = trans.r + self.learning_rate * Vs_prime
        pass

    def solve_mdp(self):
        # doesn't yet support stochastic environments, so state transition
        # probabilities are not included here
        pass

    def process(self):
        self.compute_qvalues(self.transition_buffer)

        # gather states for each leaf
        # TODO: for efficiency, we could do this on the fly as transitions are stored
        leaves = [[]]
        for transition in self.transition_buffer:
            leaf_id = self.sense_to_state_id(transition.I)
            leaves[leaf_id].append(transition)

        for leaf in leaves:
            for dimension in range(self.sense_dimensions):
                # sort q_values according to the current attribute
                leaf_q = [l.I[dimension] for l in leaf]
                sorted_q = sorted(leaf_q)

                # loop over transitions and try splitting
                max_diff = -sys.maxsize
                max_idx = 0
                for idx in range(1, len(sorted_q)):
                    first = sorted_q[:idx]
                    last = sorted_q[idx:]
                    k_stat, p_value = stats.ks_2samp(first, last)
                    print(k_stat, p_value)

                    if k_stat > max_diff and p_value < 0.05:
                        max_diff = k_stat
                        max_idx = idx
                        if k_stat > self.stopping_criterion:
                            break
        self.solve_mdp()


def main():
    # 2d sensory input, and actions are 0, 1, or 2
    tree = ContinuousUTree(2, 3)

    sense = (0, 0)
    # make up some 2d data + reward
    for x in range(-2, 2):
        for x_dot in range(-2, 2):
            if x > 0 and x_dot > 0:
                action = 0
            elif x < 0 and x_dot < 0:
                action = 2
            else:
                action = 1
            sense_prime = (x, x_dot)
            reward = action + x - x_dot
            tree.add_transition(sense, action, sense_prime, reward)

            sense = sense_prime

    tree.process()


if __name__ == "__main__":
    main()
