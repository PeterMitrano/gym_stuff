import numpy as np
import sys


class UDTree:

    def __init__(self, sense_dimensions, action_dimensions):
        self.sense_dimensions = sense_dimensions
        self.action_dimensions = action_dimensions
        self.transition_buffer_size = 1000
        self.transition_buffer = []
        self.stopping_criterion = None
        self.Q = None

    def add_transition(self, sense, action, sense_prime, reward):
        trans = (sense, action, sense_prime, reward)
        if len(self.transition_buffer) < self.transition_buffer_size:
            self.transition_buffer.append(trans)
        else:
            # replace random transition
            idx = np.random.randint(0, self.transition_buffer_size)
            self.transition_buffer[idx] = trans

    def process(self):
        print(self.transition_buffer)
        for dimension in range(self.sense_dimensions):
            # sort transitions according to the current attribute
            sorted_t = sorted(self.transition_buffer, key=lambda t: t[dimension])

            # loop over transitions and try splitting
            max_diff = -sys.maxsize
            max_idx = 0
            for idx in range(sorted_t):
                first = sorted_t[:idx]
                last = sorted_t[idx:]
                difference = self.compute_diff(first, last)

                if difference > max_diff:
                    max_diff = difference
                    max_idx = idx
                    if difference > self.stopping_criterion:
                        break



    def compute_diff(self, first, last):
        return 0



def main():
    tree = UDTree(2, 1)

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