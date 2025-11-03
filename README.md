# Evolutionary Prisoner's Dilemma Simulation

This project simulates the evolution of cooperation among autonomous agents in a spatial environment using an evolutionary Prisoner’s Dilemma framework.  
Each agent is modeled as a simple neural network that learns to cooperate or defect based on experience and evolutionary selection — demonstrating how cooperation can emerge without human-designed strategies.

---

## 🧠 Concept

The simulation is inspired by **Robert Axelrod’s** _The Evolution of Cooperation_, testing whether cooperation can evolve in artificial agents with no prior strategies.  
Agents interact with nearby neighbors, make decisions based on past encounters, reproduce based on success, and form clusters of cooperation or defection over time.

---

## ⚙️ How It Works

- **Agents:**  
  Each agent has:
  - A 3–5–1 neural network (input → hidden → output).  
  - Memory of opponents’ last 10 actions.  
  - Spatial coordinates (x, y).  
  - HP (health points) that determine survival and reproduction.

- **Interactions:**  
  - Agents play repeated Prisoner’s Dilemma games with neighbors within a fixed radius.  
  - Payoffs:
    | Interaction | Agent 1 | Agent 2 |
    |--------------|----------|----------|
    | Cooperate / Cooperate | +3 | +3 |
    | Cooperate / Defect | -1 | +5 |
    | Defect / Cooperate | +5 | -1 |
    | Defect / Defect | -4 | -4 |

- **Evolution:**  
  - Agents with higher HP may clone themselves (with mutation).  
  - Low-HP agents die off each generation.  
  - The population evolves over 50 generations (default).

- **Plots generated automatically:**  
  - Average HP per generation  
  - Cooperation rate per generation  
  - Agent population over time  
  - Cooperation distribution pie chart  

All plots are saved in the `plots/` directory.

---

---

## 🚀 How to Run

### Requirements
Install dependencies:
```bash
pip install numpy matplotlib
