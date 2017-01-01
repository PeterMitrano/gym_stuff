import numpy as np


class ObservationSpace:

    def __init__(self, dimensions):
        self.dims = dimensions
        self.steps = []

    def observation_to_index(self, observation):
        theta = (theta) // self.angle_step
        dtheta = (dtheta) // self.dtheta_step

        # this is like flattening a 2d array
        return int(theta + (self.angle_n * dtheta))

    def index_to_observation(self, state_idx):
        theta = (state_idx % self.angle_n) * self.angle_step + self.min_angle + self.epsilon
        if theta - self.epsilon <= self.max_angle < theta:
            theta -= self.epsilon
        dtheta = (state_idx // self.angle_n) * self.dtheta_step + self.min_dtheta + self.epsilon
        if dtheta - self.epsilon <= self.max_dtheta < dtheta:
            theta -= self.epsilon
        return [cos(theta), sin(theta), dtheta]


class ActionSpace:

    def index_to_action(self, action_idx):
        return action_idx * self.action_step + self.min_action

    def action_to_index(self, action):
        if action > self.max_action:
            raise ValueError("Action value %f is greater than max of %f", action, self.max_action)

        action_idx = (action - self.min_action + self.epsilon) // self.action_step
        action_idx = min(action_idx, 19)
        return int(action_idx)

