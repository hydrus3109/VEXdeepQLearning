import matplotlib.pyplot as plt
import numpy as np
import time

from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv, train_dqn_agent

def train_and_evaluate():
    # Create environment
    env = HighStakesEnv()
    
    # Set up TensorFlow for better compatibility
    import tensorflow as tf
    import os
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    
    # Choose CPU over GPU for simplicity
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    
    # Train the agent
    print("Starting training...")
    agent, scores = train_dqn_agent(env, episodes=20)  # Adjust episodes as needed
    
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

def evaluate_agent(env, agent, episodes=2):
    """Evaluate a trained agent's performance"""
    for episode in range(episodes):
        state = env.reset()
        state = env.get_observation()
        total_reward = 0
        total_score = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode+1}")
        env.render()
        time.sleep(1)
        agent.reset()
        while not done:
            # Get accessible objects
            accessible_objects = env.get_accessible_objects()
            print(accessible_objects)
            if len(accessible_objects) == 0:
                break
                
            # Choose best action (no exploration)
            agent.epsilon = 0  # Turn off exploration
            action = agent.act(state, accessible_objects)
            # Take action
            next_state, reward, done, _ , score = env.step(action)
            next_state = env.get_observation()
            agent.remember(state, action, reward, next_state, done)
            if(reward < 0):
                k = 'index'
                #print(accessible_objects)
                print(f"Action chosen: {action}")
            # Update state and score
            state = next_state
            total_score += score
            total_reward += reward
            step += 1
            
            print(f"Step {step}, Reward: {reward:.2f}, Score: {score:.2f}, Total: {total_reward:.2f}, Total Score: {total_score:.2f}")
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
