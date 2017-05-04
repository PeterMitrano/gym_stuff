import unittest
import numpy as np
from Spaces import ObservationSpace
from Spaces import ActionSpace


class ActionTest(unittest.TestCase):

    def setUp(self):
        self.a = ActionSpace

    def test_observation_space(self):
        for _ in range(1000):
            action = np.random.rand()
            idx = self.a.action_to_index(action)
            action2 = self.a.index_to_action(idx)
            self.assertEqual(action, action2)


class ObservationTest(unittest.TestCase):

    def setUp(self):
        self.o = ObservationSpace

    def test_observation_space(self):
        for _ in range(1000):
            obs = np.random.rand(d)
            idx = self.o.observation_to_index(obs)
            obs2 = self.o.index_to_observation(idx)
            self.assertEqual(obs, obs2)
