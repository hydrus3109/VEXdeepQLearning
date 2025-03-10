from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv
import unittest

class TestHighStakesEnv(unittest.TestCase):

    def setUp(self):
        self.env = HighStakesEnv()
        self.env.reset()

    def test_initial_state(self):
        state = self.env.state
        self.assertIsNotNone(state)
        self.assertEqual(state['score'], 0)
        self.assertEqual(state['robot_position'], (0, 0))

    def test_step_function(self):
        action = self.env.action_space.sample()  # Sample a random action
        next_state, reward, done, info = self.env.step(action)
        self.assertIsNotNone(next_state)
        self.assertIn(reward, [0, 1])  # Assuming the reward is either 0 or 1
        self.assertIsInstance(done, bool)

    def test_reset_function(self):
        initial_state = self.env.state
        self.env.step(self.env.action_space.sample())
        self.env.reset()
        self.assertNotEqual(initial_state, self.env.state)

    def test_render_function(self):
        self.env.render()  # Ensure render does not raise an error

    def tearDown(self):
        self.env.close()

if __name__ == '__main__':
    unittest.main()