# Frozen Lake MDP Solver

## Overview
This project is part of a **university Artificial Intelligence course**, where we implemented a **Frozen Lake** environment solver using two different **Markov Decision Process (MDP)** algorithms:
- **Value Iteration**
- **Policy Iteration**

The game environment is based on the classic **FrozenLake-v1** from OpenAI Gym, where the agent must navigate across a slippery frozen lake to reach the goal while avoiding holes. The project provides a graphical interface to select the algorithm and visualize the agent's decision-making process.

## Features
- **Two MDP-based Solvers:**
  - **Value Iteration**: Computes the optimal policy by iteratively updating state values.
  - **Policy Iteration**: Iteratively improves policy based on value function evaluation.
- **Graphical Interface (Tkinter)** for algorithm selection.
- **Heatmap Visualization** to display state values.
- **Randomized Holes** for increased difficulty in hard mode.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/frozen-lake-mdp.git
   cd frozen-lake-mdp
   ```
2. Install dependencies (not included in the repository due to size constraints):
   ```bash
   pip install numpy matplotlib gym pygame tkinter
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## How It Works
1. **Main Menu:** Select **Value Iteration** or **Policy Iteration**.
2. **MDP Computation:** The selected algorithm computes the optimal policy.
3. **Game Execution:** The agent follows the computed policy to reach the goal.
4. **Heatmap Display:** Shows state values based on the computed value function.

## Algorithms
### Value Iteration
- Iteratively updates value function for each state based on the Bellman optimality equation.
- Converges when value updates fall below a set threshold.
- Extracts the optimal policy by selecting actions with the highest Q-value.

### Policy Iteration
- Starts with an initial policy and computes the value function for that policy.
- Improves the policy iteratively until convergence.
- Ensures policy stability through repeated evaluation and improvement steps.

## File Structure
```
.
├── main.py           # Main script with GUI and algorithm selection
├── source.py         # Custom Frozen Lake environment implementation
├── requirements.txt  # Python dependencies (not included due to size)
```

## Customization
- Modify `source.py` to adjust lake size, hole placement, or transition probabilities.
- Adjust `gamma` and `threshold` in `main.py` to fine-tune MDP convergence.

## Screenshots
![Select Value Iteration](https://github.com/Mahdi-Rahbar/Frozen-Lake-MDP---AI-Project/blob/main/Screenshots/Select%20Value%20Iteration.png?raw=true)  
![Value Iteration](https://github.com/Mahdi-Rahbar/Frozen-Lake-MDP---AI-Project/blob/main/Screenshots/Value%20Iteration.gif?raw=true) 
![Select Policy Iteration](https://github.com/Mahdi-Rahbar/Frozen-Lake-MDP---AI-Project/blob/main/Screenshots/Select%20Policy%20Iteration.png?raw=true) 
![Policy Iteration](https://github.com/Mahdi-Rahbar/Frozen-Lake-MDP---AI-Project/blob/main/Screenshots/Policy%20Iteration.gif?raw=true) 

