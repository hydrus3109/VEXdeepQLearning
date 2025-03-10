class PerformanceMonitor:
    def __init__(self):
        self.total_episodes = 0
        self.total_rewards = 0
        self.successful_episodes = 0

    def reset(self):
        self.total_episodes = 0
        self.total_rewards = 0
        self.successful_episodes = 0

    def update(self, reward, success):
        self.total_rewards += reward
        self.total_episodes += 1
        if success:
            self.successful_episodes += 1

    def report(self):
        return {
            "total_episodes": self.total_episodes,
            "total_rewards": self.total_rewards,
            "successful_episodes": self.successful_episodes,
            "average_reward": self.total_rewards / self.total_episodes if self.total_episodes > 0 else 0,
        }