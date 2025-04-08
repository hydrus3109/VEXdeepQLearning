import gym
import numpy as np
import numpy
import matplotlib.pyplot as plt
import json
import os
import math
# Add required imports for DQNAgent
from collections import deque
import random
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.optimizers import Adam

class HighStakesEnv(gym.Env):
    def __init__(self):
        super(HighStakesEnv, self).__init__()
        
        # Load game configuration from JSON
        self.config_path = 'vex_high_stakes_gym/envs/assets/field_layout.json'
        self.load_config()
        
        # Calculate maximum possible actions (all rings, goals, and corners)
        max_actions = len(self.state['rings']) + len(self.state['goals']) + 4  # Rings + goals + 4 corners
        self.action_space = gym.spaces.Discrete(max_actions)
        
        # Create a sample observation to determine its shape
        sample_obs = self.get_observation()
        obs_size = len(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(obs_size,), 
            dtype=numpy.float32
        )
        
        self.score = 0
        self.time_limit = 60  # 1 minute
        self.current_time = 0

    def load_config(self):
        self.score = 0
        self.current_time = 0
        """Load field layout from JSON file and initialize state"""
        try:
            with open(self.config_path, 'r') as f:
                self.field_config = json.load(f)
                # Update time limit from config if available
                if "time_limit" in self.field_config:
                    self.time_limit = self.field_config["time_limit"]
        except FileNotFoundError:
            print(f"Warning: Config file not found at {self.config_path}. Using default configuration.")
            self.field_config = {
                "field": {
                    "dimensions": {"width": 12, "height": 6},
                    "goals": [
                        {"position": {"x": 1, "y": 5}, "size": {"width": 1, "height": 0.5}, "points": 10},
                        {"position": {"x": 11, "y": 5}, "size": {"width": 1, "height": 0.5}, "points": 10}
                    ],
                    "rings": [
                        {"position": {"x": 3, "y": 3}, "size": {"radius": 0.5}, "points": 5},
                        {"position": {"x": 9, "y": 3}, "size": {"radius": 0.5}, "points": 5}
                    ]
                },
                "time_limit": 60
            }
        
        # Initialize state based on the loaded/default configuration
        field_width = self.field_config['field']['dimensions']['width']
        field_height = self.field_config['field']['dimensions']['height']
        
        self.state = {
            # Robot starts at bottom left by default
            'robot': {
                'position': [20, 70],
                'orientation': 0,
                'has_ring': False,
                'holding_goal': False
            },
            # Initialize rings from config
            'rings': [
                {
                    'position': [ring['position']['x'], ring['position']['y']],
                    'radius': ring['size']['radius'],
                    'points': ring['points'],
                    'collected': False
                } for ring in self.field_config['field']['rings']
            ],
            # Initialize goals from config
            'goals': [
                {
                    'position': [goal['position']['x'], goal['position']['y']],
                    'scorable': goal['scorable'],
                    'size' :[2*goal['size']['radius'], 2*goal['size']['radius']],
                    'rings_scored': 0,
                    'is_held': False,
                    'is_mobile': True,
                    'in_corner': False  # Track if goal is in a corner
                } for goal in self.field_config['field']['mobile_goals']
            ],
            # Field dimensions
            'field': {
                'width': field_width,
                'height': field_height
            },
            # Add corners to the state
            'corners': [
                {'position': [0, 0], 'has_goal': False, 'goal_index': -1},
                {'position': [0, field_height], 'has_goal': False, 'goal_index': -1},
                {'position': [field_width, 0], 'has_goal': False, 'goal_index': -1},
                {'position': [field_width, field_height], 'has_goal': False, 'goal_index': -1}
            ]
        }

    def reset(self):
        self.score = 0
        self.current_time = 0
        self.load_config()  # This now also initializes the state
        return self.state

    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        reward = 0
        done = False
        info = {}
        
        # Get list of accessible objects
        accessible_objects = self.get_accessible_objects()
        # Check if action index is valid
        if action < 0 or action >= len(accessible_objects):
            reward -= 10  # Invalid action penalty
        else:
            # Get target object
            target_obj = accessible_objects[action]
            
            # Estimate movement time
            movement_time = self.estimate_movement_time(
                self.state['robot']['position'], 
                self.state['robot']['orientation'],
                target_obj['position']
            )
            
            # Apply time penalty (0.05 points per second)
            time_penalty = movement_time * 0.05
            reward -= time_penalty
            
            # Update robot position and orientation
            target_vector = [
                target_obj['position'][0] - self.state['robot']['position'][0],
                target_obj['position'][1] - self.state['robot']['position'][1]
            ]
            target_angle = math.atan2(target_vector[1], target_vector[0])
            self.state['robot']['position'] = target_obj['position']
            self.state['robot']['orientation'] = target_angle
            
            # Handle interaction based on target object type
            if target_obj['type'] == 'goal':
                # Mobile goal interaction
                if not self.state['robot']['holding_goal']:
                    # Only allow picking up a goal if it's not in a corner
                    if not self.state['goals'][target_obj['index']]['in_corner']:
                        self.state['robot']['holding_goal'] = True
                        self.state['goals'][target_obj['index']]['is_mobile'] = False  # Robot is holding it
                        self.state['goals'][target_obj['index']]['is_held'] = True
                        reward += 10  # Reward for acquiring mobile goal
                    else:
                        reward -= 5  # Penalty for trying to pick up a goal in a corner
                else:
                    reward -= 10  # Penalty for trying to grab another goal
                    
            elif target_obj['type'] == 'ring':
                # Ring interaction
                if self.state['robot']['holding_goal']:
                    goal_index = self.get_held_goal_index()
                    goal = self.state['goals'][goal_index]
                    ring_count = goal['rings_scored']
                    
                    if ring_count < 6:  # Goal not full
                        self.state['rings'][target_obj['index']]['collected'] = True
                        goal['rings_scored'] += 1
                        
                        if ring_count == 0:
                            reward += 3  # First ring bonus
                        else:
                            reward += 1  # Regular ring score
                    else:
                        reward -= 1  # Penalty for full goal
                else:
                    reward -= 5  # No goal clamped
                    
            elif target_obj['type'] == 'corner':
                # Corner interaction
                if self.state['robot']['holding_goal']:
                    corner_index = target_obj['index']
                    goal_index = self.get_held_goal_index()
                    
                    # Place goal in corner
                    self.state['robot']['holding_goal'] = False
                    self.state['corners'][corner_index]['has_goal'] = True
                    self.state['corners'][corner_index]['goal_index'] = goal_index
                    
                    # Update goal position and state
                    goal = self.state['goals'][goal_index]
                    goal['position'] = list(self.state['corners'][corner_index]['position'])
                    goal['is_mobile'] = False
                    goal['is_held'] = False
                    goal['in_corner'] = True  # Mark as in a corner
                    
                    reward += 5  # Score for placing goal in corner
                else:
                    reward -= 1  # No goal to place
            
        # Update time
        self.current_time += 1
        
        # Check if time limit reached
        if self.current_time >= self.time_limit:
            done = True
        
        # Update score
        self.score += reward
        
        # Update observation
        observation = self.get_observation()
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        Render the current state of the High Stakes environment.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import Affine2D
        import numpy as np
        
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()  # Interactive mode
        
        # Clear previous render
        self.ax.clear()
        
        # Set up field boundaries (assuming standard VEX field)
        field_width, field_height = 144, 144  # Standard VEX field dimensions
        self.ax.set_xlim(0, field_width)
        self.ax.set_ylim(0, field_height)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        # Draw field boundary
        self.ax.add_patch(plt.Rectangle((0, 0), field_width, field_height, 
                                      fill=False, edgecolor='black', linewidth=2))
        
        # Draw game elements based on state
        if hasattr(self, 'state'):
            # Draw robot position with correct orientation
            if 'robot' in self.state:
                x, y = self.state['robot']['position']
                orientation_rad = self.state['robot']['orientation']
                
                # Use transform to properly rotate rectangle around its center
                robot_size = 16
                
                # Create a rectangle centered at origin (0,0)
                rect = Rectangle((-robot_size/2, -robot_size/2), robot_size, robot_size, 
                                color='green', alpha=0.7)
                
                # Create transformation: first rotate around origin, then translate to (x,y)
                t = Affine2D().rotate(orientation_rad).translate(x, y)
                rect.set_transform(t + self.ax.transData)
                self.ax.add_patch(rect)
                
                # Draw direction indicator
                arrow_length = robot_size * 0.75
                dir_x = x + np.cos(orientation_rad) * arrow_length
                dir_y = y + np.sin(orientation_rad) * arrow_length
                self.ax.plot([x, dir_x], [y, dir_y], color='black', linewidth=3)
                self.ax.plot([dir_x], [dir_y], 'o', color='black', markersize=6)
                
                # Show if robot is holding a goal
                if self.state['robot']['holding_goal']:
                    self.ax.text(x, y, "Has Goal", ha='center', va='center', 
                               color='white', fontsize=8, weight='bold', zorder=10)
            
            # Draw rings (only uncollected ones)
            if 'rings' in self.state:
                for ring in self.state['rings']:
                    if not ring['collected']:
                        x, y = ring['position']
                        self.ax.add_patch(plt.Circle((x, y), 3.5, color='red', alpha=0.8))
            
            # Draw regular goals (not held by robot and not in corners)
            if 'goals' in self.state:
                for i, goal in enumerate(self.state['goals']):
                    if not goal['is_held'] and not goal['in_corner']:
                        x, y = goal['position']
                        self.ax.add_patch(plt.Rectangle((x-5, y-5), 10, 10, color='blue', alpha=0.6))
                        self.ax.text(x, y, f"Goal {i}", ha='center', va='center', 
                                   fontsize=8, color='white', weight='bold')
                        if goal['rings_scored'] > 0:
                            self.ax.text(x, y-3, f"{goal['rings_scored']} rings", 
                                       ha='center', va='center', fontsize=7, color='white')
            
            # Draw corners and goals in corners
            if 'corners' in self.state:
                for i, corner in enumerate(self.state['corners']):
                    cx, cy = corner['position']
                    
                    # If a goal is in this corner, draw it
                    if corner['has_goal'] and corner['goal_index'] >= 0:
                        goal = self.state['goals'][corner['goal_index']]
                        self.ax.add_patch(plt.Rectangle((cx, cy), 12, 12, 
                                                      color='purple', alpha=0.7))
                        # Show rings scored
                        if goal['rings_scored'] > 0:
                            self.ax.text(cx, cy, f"{goal['rings_scored']} rings", 
                                       ha='center', va='center', fontsize=8, 
                                       color='black', weight='bold')
        
        # Add title with score and time
        self.ax.set_title(f'Score: {self.score:.1f}, Time: {self.current_time}/{self.time_limit}')
        
    
        
        self.fig.canvas.draw()
        
        plt.savefig('foo.png')
        if mode == 'human':
            plt.pause(0.1)  # Small pause to display
            return None
        elif mode == 'rgb_array':
            # Convert canvas to RGB array
            data = numpy.frombuffer(self.fig.canvas.tostring_rgb(), dtype=numpy.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        # Clean up resources if needed
        pass
    
    def check_collision(self, start_pos, end_pos, width=2.5):
        """Check if a straight line path between two points collides with any objects"""
        # Vector from start to end
        vec = numpy.array([end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]])
        vec_length = numpy.linalg.norm(vec)
        if vec_length == 0:
            return False
            
        # Normalize vector
        vec = vec / vec_length
        
        # Check all objects (rings, goals, corners)
        all_objects = []
        
        # Add rings
        for ring in self.state['rings']:
            if not ring['collected']:
                all_objects.append({
                    'position': ring['position'],
                    'radius': ring['radius'],
                    'type': 'ring'
                })
                
        # Add goals
        for goal in self.state['goals']:
            all_objects.append({
                'position': goal['position'],
                'radius': max(goal['size'][0], goal['size'][1]),  # Use larger dimension
                'type': 'goal'
            })
                
        # Add corners (field corners)
        field_width = self.state['field']['width']
        field_height = self.state['field']['height']
        corner_radius = 1.0  # Assume corners have some radius
        corners = [
            {'position': [0, 0], 'radius': corner_radius, 'type': 'corner'},
            {'position': [field_width, 0], 'radius': corner_radius, 'type': 'corner'},
            {'position': [0, field_height], 'radius': corner_radius, 'type': 'corner'},
            {'position': [field_width, field_height], 'radius': corner_radius, 'type': 'corner'}
        ]
        all_objects.extend(corners)
        
        # Check collision with each object
        for obj in all_objects:
            # Skip the start and end objects
            if (numpy.allclose(start_pos, obj['position']) or 
                numpy.allclose(end_pos, obj['position'])):
                continue
                
            # Compute closest point on line to object
            obj_pos = numpy.array(obj['position'])
            start_to_obj = obj_pos - numpy.array(start_pos)
            projection = numpy.dot(start_to_obj, vec)
            
            # If projection is outside line segment, no collision
            if projection < 0 or projection > vec_length:
                continue
                
            # Closest point on line
            closest_point = numpy.array(start_pos) + projection * vec
            
            # Distance from closest point to object center
            distance = numpy.linalg.norm(closest_point - obj_pos)
            
            # Check if distance is less than sum of object radius and path width
            if distance < (obj['radius'] + width/2):
                return True
                
        return False
    
    def estimate_movement_time(self, start_pos, start_orient, end_pos):
        """Estimate time to reach target position"""
        # Calculate turn angle
        target_vector = [end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]]
        target_angle = math.atan2(target_vector[1], target_vector[0])
        angle_diff = abs((target_angle - start_orient) % (2 * math.pi))
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
            
        # Time for turning (1.5 seconds for full 2π rotation)
        turn_time = (angle_diff / (2 * math.pi)) * 1.5
        
        # Distance to target
        distance = math.sqrt(target_vector[0]**2 + target_vector[1]**2)
        
        # Time for driving (1.5 seconds per 12 inches)
        drive_time = (distance / 12) * 1.5
        
        return turn_time + drive_time
    
    def get_accessible_objects(self):
        """Get list of objects that robot can move to without collisions"""
        accessible_objects = []
        robot_pos = self.state['robot']['position']
        
        # Check rings
        for i, ring in enumerate(self.state['rings']):
            if not ring['collected'] and not self.check_collision(robot_pos, ring['position']):
                accessible_objects.append({
                    'index': i,
                    'position': ring['position'],
                    'type': 'ring'
                })
        
        # Check goals - don't include goals in corners or being held
        for i, goal in enumerate(self.state['goals']):
            if not goal['in_corner'] and not goal['is_held'] and not self.check_collision(robot_pos, goal['position']):
                accessible_objects.append({
                    'index': i,
                    'position': goal['position'],
                    'type': 'goal'
                })
        
        # Add corners
        for i, corner in enumerate(self.state['corners']):
            #print(corner['position'], corner['has_goal'])
            if not self.check_collision(robot_pos, corner['position']) and not corner['has_goal']:
                accessible_objects.append({
                    'position': corner['position'],
                    'type': 'corner',
                    'index': i
                })
        
        return accessible_objects
    

    def get_observation(self):
        """Convert state to observation for the agent"""
        # This is a simplified observation vector - you might want to expand this
        robot_pos = self.state['robot']['position']
        robot_orient = self.state['robot']['orientation']
        holding_goal = 1.0 if self.state['robot']['holding_goal'] else 0.0
        
        # Create a flat observation vector
        obs = [robot_pos[0], robot_pos[1], robot_orient, holding_goal]
        
        # Add rings information
        for ring in self.state['rings']:
            if not ring['collected']:
                obs.extend([ring['position'][0], ring['position'][1], 1.0])  # Position + availability flag
            else:
                obs.extend([0.0, 0.0, 0.0])  # Placeholder for collected rings
        
        # Add goals information
        for goal in self.state['goals']:
            obs.extend([
                goal['position'][0], goal['position'][1], 
                float(goal['rings_scored']),
                1.0 if goal['is_mobile'] else 0.0
            ])
            
        return numpy.array(obs, dtype=numpy.float32)
    
    def get_held_goal_index(self):
        """Find the index of the goal being held by the robot"""
        for i, goal in enumerate(self.state['goals']):
            if not goal['is_mobile'] and self.state['robot']['holding_goal']:
                return i
        return -1  # Not holding any goal

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        """Neural network for deep Q learning"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer= 'adam', metrics=['accuracy'])
        return model
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, accessible_objects_count):
        """Select action based on epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(accessible_objects_count)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0][:accessible_objects_count])
        
    def replay(self, batch_size):
        """Train on batches of experiences"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Get next state prediction and max Q value for it
                print(next_state.shape)
                print(next_state.reshape(1, -1).shape)
                print(next_state)
                next_state_pred = self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target += self.gamma * np.amax(next_state_pred)
            
            # Get current Q values and update the target for action
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            
            # Train the network
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training code
def train_dqn_agent(env, episodes=1000):
    """
    Train a DQN agent on the High Stakes environment
    
    Args:
        env: The environment instance
        episodes: Number of episodes to train for
        
    Returns:
        agent: Trained DQN agent
        scores: List of scores for each episode
    """
    # Get initial state to determine observation size
    state = env.reset()
    obs = env.get_observation()
    state_size = len(obs)
    
    # Maximum possible actions (worst case - all objects accessible)
    max_actions = len(env.state['rings']) + len(env.state['goals']) + 4  # Rings + goals + 4 corners
    
    # Create agent with observation size and max action size
    agent = DQNAgent(state_size, max_actions)
    
    # Training loop
    batch_size = 32
    scores = []
    
    for e in range(episodes):
        # Reset environment for new episode
        env.reset()
        state = env.get_observation()
        total_reward = 0
        done = False
        
        # Episode loop
        while not done:
            # Get accessible objects
            accessible_objects = env.get_accessible_objects()
            if len(accessible_objects) == 0:
                break
                
            # Choose action
            action = agent.act(state, len(accessible_objects))
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = env.get_observation()
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            total_reward += reward
            
            # Visualize training (optional, can be commented out for faster training)
            # env.render()
            
        # Train the agent after episode completion
        agent.replay(batch_size)
        
        # Track progress
        scores.append(total_reward)
        
        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return agent, scores
