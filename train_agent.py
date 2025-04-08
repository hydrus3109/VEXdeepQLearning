import matplotlib.pyplot as plt
import numpy as np
import time

from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv, train_dqn_agent

def train_and_evaluate():
    # Create environment
    env = HighStakesEnv()
    
    # Train the agent
    print("Starting training...")
    agent, scores = train_dqn_agent(env, episodes=100)  # Adjust episodes as needed
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    plt.savefig('training_progress.png')
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    evaluate_agent(env, agent)
    
    return agent

def evaluate_agent(env, agent, episodes=5):
    """Evaluate a trained agent's performance"""
    for episode in range(episodes):
        state = env.reset()
        state = env.get_observation()
        total_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode+1}")
        env.render()
        time.sleep(1)
        
        while not done:
            # Get accessible objects
            accessible_objects = env.get_accessible_objects()
            if len(accessible_objects) == 0:
                break
                
            # Choose best action (no exploration)
            agent.epsilon = 0  # Turn off exploration
            action = agent.act(state, len(accessible_objects))
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = env.get_observation()
            
            # Update state and score
            state = next_state
            total_reward += reward
            step += 1
            
            print(f"Step {step}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            env.render()
            time.sleep(0.5)  # Slow down for visualization
            
            if done:
                break
        
        print(f"Episode {episode+1} finished with score: {total_reward:.2f}")
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep plot open at the end

if __name__ == "__main__":
    trained_agent = train_and_evaluate()
    print("Training and evaluation complete!")
