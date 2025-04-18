import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import time
import tensorflow as tf

# Import our environment and agent
from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv, DQNAgent

def create_training_timelapse(episodes=10, frames_per_episode=10, output_path="training_timelapse.mp4"):
    """
    Creates a timelapse video of the training progress.
    
    Args:
        episodes: Number of episodes to train for
        frames_per_episode: Number of frames to capture per episode
        output_path: Path to save the output video
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(episodes)
    print(frames_per_episode)
    # Initialize environment
    env = HighStakesEnv()
    
    # Get initial state to determine observation size
    env.reset()
    obs = env.get_observation()
    state_size = len(obs)
    
    # Maximum possible actions
    max_actions = len(env.state['rings']) + len(env.state['goals']) + 4  # Rings + goals + 4 corners
    
    # Create agent with observation size and max action size
    agent = DQNAgent(state_size, max_actions)
    
    # Training parameters
    batch_size = 32
    
    # Keep track of scores for plotting
    all_scores = []
    all_rewards = []
    
    # For video creation
    frames = []
    
    # Setup figure for rendering
    plt.figure(figsize=(10, 8))
    
    print("Starting training with video capture...")
    
    for e in tqdm(range(episodes)):
        # Reset environment and agent for new episode
        render = False
        if(e % 5 == 0):
            render = True
        env.reset()
        agent.reset()
        
        state = env.get_observation()
        total_reward = 0
        total_score = 0
        done = False
        step_count = 0

        frames_this_episode = 0
        
        # Episode loop
        while not done:
            if render:
            # Capture frame if it's time (or if it's the last frame of the episode)
                if frames_this_episode < frames_per_episode or done:
                    env.render(mode='human')  # Update the plot
                    
                    # Add episode info as text to the frame
                    plt.figtext(0.05, 0.02, f"Episode: {e+1}/{episodes}, Reward: {total_reward:.1f}, Score: {total_score}", 
                            backgroundcolor='white', alpha=0.7)
                    plt.figtext(0.05, 0.06, f"Epsilon: {agent.epsilon:.2f} (exploration rate)", 
                            backgroundcolor='white', alpha=0.7)
                    
                    # Convert plot to image
                    fig = plt.gcf()
                    fig.canvas.draw()
                    frame = np.array(fig.canvas.renderer.buffer_rgba())
                    frames.append(frame)
                    if(frames_this_episode == 0):
                        # Add a copy of the first frame to ensure we have enough frames
                        frames.append(frame.copy())
                        frames.append(frame.copy())
                        frames.append(frame.copy())
                        frames.append(frame.copy())
                    if(frames_this_episode == frames_per_episode - 1):
                        # Add a copy of the last frame to ensure we have enough frames
                        frames.append(frame.copy())
                        frames.append(frame.copy())
                        frames.append(frame.copy())
                        frames.append(frame.copy())
                    frames.append(frame.copy())
                    frames.append(frame.copy())
                    frames_this_episode += 1
                            # Get accessible objects

            accessible_objects = env.get_accessible_objects()
            if len(accessible_objects) == 0:
                break
                
            # Choose action
            action = agent.act(state, accessible_objects)
            if action is None:  # No valid actions available
                break
                
            # Take action
            next_state, reward, done, _, score = env.step(action)
            next_state = env.get_observation()
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and scores
            state = next_state
            total_reward += reward
            total_score += score
            step_count += 1
        # Train the agent after episode completion
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # Track progress
        all_rewards.append(total_reward)
        all_scores.append(total_score)
        
        # Print progress every 10 episodes
        if (e + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            print(f"Episode: {e+1}/{episodes}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    print("Training complete. Creating video...")
    
    # Create video from frames
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 8, (width, height))
    
    for frame in frames:
        # Convert RGBA to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        video.write(frame_bgr)
    
    video.release()
    print(f"Video saved to {output_path}")
    
    # Plot final learning curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(all_rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(all_scores)
    plt.title('Scores per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Score')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Learning curves saved to learning_curves.png")
    
    return agent, all_rewards

if __name__ == "__main__":
    # Set up TensorFlow to avoid warnings
    physical_devices = tf.config.list_physical_devices('CPU')

    # Create timelapse with reasonable defaults
    # You can adjust these parameters as needed
    agent, rewards = create_training_timelapse(
        episodes=70,# Fewer episodes for quicker demonstration
        frames_per_episode=10,  # Number of frames to capture in each episode
        output_path="training_timelapse.mp4"
    )
    
    print(f"Final average reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}")
