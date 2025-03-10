from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv
import numpy as np
import torch

class TrainedAgent:
    def __init__(self, model_path):
        self.env = HighStakesEnv()
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        model.eval()
        return model

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state_tensor).argmax().item()
        return action

    def run(self, duration=60):
        state = self.env.reset()
        total_reward = 0
        for _ in range(duration):
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                state = self.env.reset()
        print(f'Total Reward: {total_reward}')

if __name__ == "__main__":
    agent = TrainedAgent(model_path='path/to/your/model.pth')
    agent.run()