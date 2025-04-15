import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np

from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv

def run_manual_simulation():
    env = HighStakesEnv()
    env.reset()
    env.load_config()
    
    # Total reward tracking
    total_reward = 0
    total_score = 0
    
    # Display initial state
    print("Initial State")
    env.render()
    time.sleep(1)
    
    # Sequence of actions to demonstrate
    print("\n--- Starting Manual Simulation ---")
    
    # 1. Move to a goal
    print("\nStep 1: Moving to grab a mobile goal")
    accessible_objects = env.get_accessible_objects()
    print(accessible_objects)
    goal_actions = [i for i, obj in enumerate(accessible_objects) if obj['type'] == 'goal']
    
    if goal_actions:
        action = goal_actions[0]  # First accessible goal
        obs, reward, done, _, score = env.step(action)
        total_reward += reward
        total_score += score
        print(f"Moved to goal, reward: {reward}, Score: {total_score}")
        env.render()
        time.sleep(1)
    
    # 2. Move to a ring with the goal
    print("\nStep 2: Moving to a ring with the goal")
    accessible_objects = env.get_accessible_objects()
    ring_actions = [i for i, obj in enumerate(accessible_objects) if obj['type'] == 'ring']
    
    if ring_actions:
        action = ring_actions[0]  # First accessible ring
        obs, reward, done, _, score = env.step(action)
        total_reward += reward
        total_score += score
        print(f"Moved to goal, reward: {reward}, Score: {total_score}")
        env.render()
        time.sleep(1)
    
    # 3. Move to another ring
    print("\nStep 3: Moving to another ring")
    accessible_objects = env.get_accessible_objects()
    ring_actions = [i for i, obj in enumerate(accessible_objects) if obj['type'] == 'ring']
    
    if len(ring_actions) > 0:
        action = ring_actions[0]  # Another accessible ring
        obs, reward, done, _,score = env.step(action)
        total_reward += reward
        total_score += score
        print(f"Moved to goal, reward: {reward}, Score: {total_score}")
        env.render()
        time.sleep(1)
    
    # 4. Move to a corner to place the goal
    print("\nStep 4: Moving to a corner to place goal")
    accessible_objects = env.get_accessible_objects()
    corner_actions = [i for i, obj in enumerate(accessible_objects) if obj['type'] == 'corner']
    
    if corner_actions:
        action = corner_actions[0]  # First accessible corner
        obs, reward, done, _, score = env.step(action)
        total_reward += reward
        total_score += score
        print(f"Moved to goal, reward: {reward}, Score: {total_score}")
        env.render()
        time.sleep(1)
    
    # 5. Move to another goal
    print("\nStep 5: Moving to grab another goal")
    accessible_objects = env.get_accessible_objects()
    goal_actions = [i for i, obj in enumerate(accessible_objects) if obj['type'] == 'goal']
    
    if goal_actions:
        action = goal_actions[0]  # Another accessible goal
        obs, reward, done, _, score = env.step(action)
        total_reward += reward
        total_score += score
        print(f"Moved to goal, reward: {reward}, Score: {total_score}")
        env.render()
        time.sleep(1)
    
    # 6. Move to another corner
    print("\nStep 6: Moving to another corner")
    accessible_objects = env.get_accessible_objects()
    #print(accessible_objects)
    corner_actions = [i for i, obj in enumerate(accessible_objects) if obj['type'] == 'corner']
    
    if corner_actions:
        action = corner_actions[0]  # Another accessible corner
        obs, reward, done, _, score = env.step(action)
        total_reward += reward
        total_score += score
        print(f"Moved to goal, reward: {reward}, Score: {total_score}")
        env.render()
        time.sleep(1)
    
    print(f"\nSimulation completed with total reward: {total_reward}")
    
    # Keep the plot open
    plt.ioff()  # Turn off interactive mode for final display
    plt.show()

if __name__ == '__main__':
    run_manual_simulation()
