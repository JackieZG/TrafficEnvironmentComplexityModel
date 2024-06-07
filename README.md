# Traffic Environment Complexity Calculation Model

This repository contains the code and documentation for my undergraduate thesis on the traffic environment complexity calculation model. The thesis discusses the importance of traffic environment complexity in autonomous vehicle testing and safety evaluation, and proposes a novel method for calculating both static and dynamic traffic environment complexity.

## Thesis Abstract

[Include the abstract from your thesis here]

## Repository Structure

- `code/`: Contains the source code for the complexity calculation models and RL environment.
  - `static_complexity.py`: Static complexity calculation code.
  - `dynamic_complexity.py`: Dynamic complexity calculation code.
  - `traffic_env.py`: Custom traffic environment for reinforcement learning.
  - `train_rl_model.py`: Script to train RL model using TF-Agents.
- `data/`: Contains sample data used in the models.
- `docs/`: Contains the thesis and related documentation.

## Getting Started

### Prerequisites

1. Ensure you have Python 3.6+ installed.
2. Install the necessary Python packages:
    ```bash
    pip install tensorflow tf-agents gym numpy matplotlib
    ```

### Static Complexity Calculation

1. To run the static complexity calculation, use the following command:
    ```bash
    python code/static_complexity.py
    ```

### Dynamic Complexity Calculation

1. To run the dynamic complexity calculation, use the following command:
    ```bash
    python code/dynamic_complexity.py
    ```

### Reinforcement Learning Environment

1. To test the custom traffic environment, use the following command:
    ```bash
    python code/traffic_env.py
    ```

### Training the Reinforcement Learning Model

1. To train the RL model using TF-Agents, use the following command:
    ```bash
    python code/train_rl_model.py
    ```

### Project Structure

- **`static_complexity.py`**: This script calculates the static complexity of the traffic environment using information entropy and grey relational analysis.
- **`dynamic_complexity.py`**: This script calculates the dynamic complexity of the traffic environment using a gravitational model to analyze dynamic elements.
- **`traffic_env.py`**: This script defines a custom Gym environment for the traffic scenario, incorporating both static and dynamic complexity calculations.
- **`train_rl_model.py`**: This script uses TF-Agents to train a reinforcement learning agent in the custom traffic environment.

### Results Visualization

1. After training the RL model, the script will visualize the training results using matplotlib.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
