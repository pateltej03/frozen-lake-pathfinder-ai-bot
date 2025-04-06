# 🧊 Solving Frozen Lake with Model-Based Reinforcement Learning

This project is a full implementation of a **model-based reinforcement learning (RL)** agent to solve the classic **Frozen Lake** environment using the [Gymnasium](https://gymnasium.farama.org/) library.

I built this as part of my Artificial Intelligence course at Penn State, but my goal went beyond just completing an assignment — I wanted to deeply understand how agents learn and plan in environments with uncertainty and sparse rewards.

---

## 📌 Overview

The Frozen Lake environment is a gridworld where the agent must reach a goal while avoiding holes in the ice. Movement is slippery, so actions don’t always lead where expected.

This project tackles the problem by:

1. **Executing a random policy for 1000 episodes** to explore the environment.
2. **Estimating** the environment’s:
    - Transition probabilities: `T(s' | s, a)`
    - Reward function: `R(s, a, s')`
3. **Running Value Iteration** to compute the optimal value function.
4. **Extracting the optimal policy** from the value function.
5. **Executing the learned policy** in the environment to solve the lake.

---

## 🧠 Key Concepts Used

-   **Markov Decision Processes (MDPs)**
-   **Value Iteration Algorithm**
-   **Policy Extraction**
-   **Manhattan Distance Heuristic** for smarter policy refinement
-   Exploration of **stochastic dynamics** due to slipperiness

---

## 🛠 Tech Stack

-   Python 3.9
-   [Gymnasium](https://gymnasium.farama.org/)
-   NumPy

---

## 🚀 How to Run

1. Install the required package:

    ```bash
    pip install gymnasium

    ```

2. Run the solution script:
   python3 solution.py

---

## 📊 Sample Output

Optimal Policy (4x4 Grid):
1 1 1 1
2 0 2 0
2 2 2 0
0 3 2 0

Legend (actions):

-   0 = LEFT
-   1 = DOWN
-   2 = RIGHT
-   3 = UP

The numbers represent the **best action to take from each grid cell**, forming a complete optimal policy to reach the goal while minimizing the chance of falling into holes.

---

## 🧩 Personal Takeaways

This project gave me hands-on experience with:

-   Deriving MDP parameters from limited data
-   Planning in uncertain, stochastic environments
-   Balancing theory and implementation in RL
-   Making intelligent decisions from sparse feedback

I also enhanced my understanding by experimenting with heuristics like **Manhattan distance** to guide the policy when multiple actions seem equally valuable — helping the agent “think ahead” spatially.

---

## 🌟 What's Next?

I'm continuing to explore RL and plan to expand this project by experimenting with:

-   **Policy Iteration**
-   **Monte Carlo methods**
-   **Q-learning (model-free)**
-   **Larger maps and dynamic environments**

---

## 🧠 Let’s Connect!

**Tej Jaideep Patel**  
B.S. Computer Engineering  
📍 Penn State University  
✉️ tejpatelce@gmail.com  
📞 814-826-5544

---
