# VEX High Stakes Gym

Welcome to the VEX High Stakes Gym project! This project provides an OpenAI Gym environment for the VEX game "High Stakes," where a robot agent can dynamically move, interact with rings and goals, and maximize points scored within a minute.

## Project Structure

The project is organized as follows:

```
vex-high-stakes-gym
├── vex_high_stakes_gym
│   ├── __init__.py
│   ├── envs
│   │   ├── __init__.py
│   │   ├── high_stakes_env.py
│   │   └── assets
│   │       └── field_layout.json
│   ├── utils
│   │   ├── __init__.py
│   │   ├── physics.py
│   │   └── rendering.py
│   └── wrappers
│       ├── __init__.py
│       └── monitoring.py
├── examples
│   ├── __init__.py
│   ├── random_agent.py
│   └── trained_agent.py
├── tests
│   ├── __init__.py
│   └── test_env.py
├── setup.py
└── requirements.txt
```

## Installation

To install the VEX High Stakes Gym, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd vex-high-stakes-gym
pip install -r requirements.txt
```

## Usage

To use the VEX High Stakes Gym environment, you can create an instance of the `HighStakesEnv` class from the `vex_high_stakes_gym.envs.high_stakes_env` module. Here is a simple example:

```python
import gym
from vex_high_stakes_gym.envs.high_stakes_env import HighStakesEnv

env = HighStakesEnv()
env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Sample random action
    env.step(action)  # Take a step in the environment
    env.render()  # Render the environment
```

## Testing

To run the tests for the environment, navigate to the project directory and execute:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.