from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv
import random

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self):
        return self.env.action_space.sample()

def main():
    env = HighStakesEnv()
    agent = RandomAgent(env)
    total_reward = 0
    done = False
    state = env.reset()

    for _ in range(60):  # Run for 60 steps (1 minute)
        action = agent.select_action()
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()

        if done:
            break

    print(f'Total Reward: {total_reward}')
    env.close()

if __name__ == "__main__":
    main()