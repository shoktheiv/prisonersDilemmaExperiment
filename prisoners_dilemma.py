import numpy as np
import matplotlib.pyplot as plt
import os
import random

NUMBER_OF_AGENTS = 100
STARTING_HP = 100
HP_DECAY = 0
DUPLICATION_THRESHOLD = 100
MAX_AGENTS = 200

TOTAL_GAMES = 1000000

class Agent:
    def __init__(self):
        # Neural network with 3 inputs (bias, own HP normalized, opponent last move), 1 hidden layer with 5 neurons, and 1 output neuron
        self.w1 = np.random.randn(5, 3)  # weights from input to hidden
        self.b1 = np.random.randn(5, 1)  # biases for hidden layer
        self.w2 = np.random.randn(1, 5)  # weights from hidden to output
        self.b2 = np.random.randn(1, 1)  # bias for output
        self.hp = STARTING_HP
        self.last_action = None  # 1 for cooperate, 0 for defect
        self.memory = {}  # opponent_id -> list of last 10 actions of opponent
        # Position for spatial interactions
        self.x = random.uniform(-50, 50)
        self.y = random.uniform(-50, 50)

    def forward(self, x):
        # x is input vector shape (3, 1)
        z1 = self.w1 @ x + self.b1  # (5,1)
        a1 = np.tanh(z1)            # activation hidden layer
        z2 = self.w2 @ a1 + self.b2 # (1,1)
        output = 1 / (1 + np.exp(-z2))  # sigmoid output
        return output

    def decide(self, opponent_id):
        # Input vector: bias=1, own HP normalized, opponent last move (average of last 10 moves, default 1)
        opponent_moves = self.memory.get(opponent_id, [1]*10)
        if len(opponent_moves) < 10:
            # pad with 1s if less than 10 moves
            opponent_moves = [1]*(10 - len(opponent_moves)) + opponent_moves
        opponent_avg = sum(opponent_moves[-10:]) / 10
        own_hp_normalized = self.hp / STARTING_HP
        x = np.array([[1], [own_hp_normalized], [opponent_avg]])
        prob_cooperate = self.forward(x)[0,0]
        action = 1 if prob_cooperate > 0.5 else 0
        self.last_action = action
        return action

class Simulation:
    def __init__(self, num_agents=NUMBER_OF_AGENTS):
        self.agents = [Agent() for _ in range(num_agents)]
        self.num_agents = num_agents
        self.history = []  # list of tuples (avg_hp, coop_rate)
        self.num_agents_history = []
        self.generation = 0
        self.generation_history = []

    def run_round(self):
        # Each pair of agents plays a random number of repeated games
        num_games_per_pair = random.randrange(40, 70)

        # Reset last_action before interactions for statistics
        for agent in self.agents:
            agent.last_action = None

        # Spatially local interactions
        interaction_radius = 20  # adjustable cluster radius
        for i, agent1 in enumerate(self.agents):
            for j, agent2 in enumerate(self.agents):
                if i == j:
                    continue
                dist_sq = (agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2
                if dist_sq > interaction_radius ** 2:
                    continue
                for _ in range(num_games_per_pair):
                    a1 = agent1.decide(j)
                    a2 = agent2.decide(i)
                    # Apply more punishing Prisoner's Dilemma payoff
                    if a1 == 1 and a2 == 1:
                        # both cooperate: small reward
                        agent1.hp += 3
                        agent2.hp += 3
                    elif a1 == 1 and a2 == 0:
                        # agent1 cooperates, agent2 defects: harsh penalty for cooperator
                        agent2.hp += 5
                        agent1.hp -= 1
                    elif a1 == 0 and a2 == 1:
                        # agent1 defects, agent2 cooperates
                        agent1.hp += 5
                        agent2.hp -= 1
                    else:
                        # both defect: moderate penalty
                        agent1.hp -= 4
                        agent2.hp -= 4
                    # Store opponent's last move in memory, keep only last 10 moves
                    agent1.memory.setdefault(j, [])
                    agent1.memory[j].append(a2)
                    if len(agent1.memory[j]) > 10:
                        agent1.memory[j].pop(0)
                    agent2.memory.setdefault(i, [])
                    agent2.memory[i].append(a1)
                    if len(agent2.memory[i]) > 10:
                        agent2.memory[i].pop(0)
                    # Update last_action for statistics tracking
                    agent1.last_action = a1
                    agent2.last_action = a2

        # Adaptive HP decay for all agents
        adaptive_decay = max(1, int(4 * (self.num_agents / MAX_AGENTS)))
        for agent in self.agents:
            agent.hp -= adaptive_decay
            if agent.hp < 0:
                agent.hp = 0

        # Remove dead agents immediately
        self.agents = [agent for agent in self.agents if agent.hp > 0]
        self.num_agents = len(self.agents)

        # Small random movement for each agent to create dynamic clusters
        for agent in self.agents:
            agent.x += random.uniform(-2, 2)
            agent.y += random.uniform(-2, 2)

        # Duplicate top 10% agents optionally based on their neural network output
        alive_agents = [agent for agent in self.agents if agent.hp > 0]
        top_10_percent_count = max(1, int(len(alive_agents) * 0.1))
        top_agents = sorted(alive_agents, key=lambda a: a.hp, reverse=True)[:top_10_percent_count]
        new_agents = []
        for agent in top_agents:
            breed_input = np.array([[1], [agent.hp / STARTING_HP], [0]])
            breed_prob = agent.forward(breed_input)[0,0]
            if random.random() < breed_prob and agent.hp > DUPLICATION_THRESHOLD:
                clone = Agent()
                clone.w1 = np.copy(agent.w1) + np.random.normal(0, 0.01, agent.w1.shape)
                clone.b1 = np.copy(agent.b1) + np.random.normal(0, 0.01, agent.b1.shape)
                clone.w2 = np.copy(agent.w2) + np.random.normal(0, 0.01, agent.w2.shape)
                clone.b2 = np.copy(agent.b2) + np.random.normal(0, 0.01, agent.b2.shape)
                # Partial memory inheritance
                clone.memory = {k: v[-5:] for k, v in agent.memory.items()}
                # Spawn clone near parent in position
                clone.x = agent.x + random.uniform(-5, 5)
                clone.y = agent.y + random.uniform(-5, 5)
                new_agents.append(clone)
        self.agents.extend(new_agents)
        self.num_agents = len(self.agents)

        # Calculate per-generation statistics using only alive agents
        alive_agents = [agent for agent in self.agents if agent.hp > 0]
        if alive_agents:
            avg_hp = np.mean([agent.hp for agent in alive_agents])
            coop_count = sum(agent.last_action if agent.last_action is not None else 0 for agent in alive_agents)
            coop_rate = coop_count / len(alive_agents)
        else:
            avg_hp = 0
            coop_rate = 0

        self.history.append((avg_hp, coop_rate))
        self.num_agents_history.append(self.num_agents)
        self.generation += 1
        self.generation_history.append(self.generation)

        # Return number of interactions (games) played this round
        return self.num_agents * (self.num_agents - 1) // 2 * num_games_per_pair

    def summary(self, round_num):
        if self.num_agents == 0:
            print("No agents remaining")
            return
        avg_hp, coop_rate = self.history[-1]
        print(f"Round {round_num}: Average HP = {avg_hp:.2f}, Cooperation Rate = {coop_rate:.2f}, Agents Left = {self.num_agents}")

    def plot_history(self):
        if not os.path.exists('plots'):
            os.makedirs('plots')
        generations = self.generation_history
        avg_hp = [h[0] for h in self.history]
        coop_rate = [h[1] for h in self.history]
        num_agents = self.num_agents_history

        # If only one generation, create new plots
        if len(generations) == 1:
            plt.figure()
            plt.plot(generations, avg_hp, label='Average HP', marker='o')
            plt.xlabel('Generation Number')
            plt.ylabel('Average HP')
            plt.title('Average HP over Generations')
            plt.legend()
            plt.savefig('plots/average_hp.png')
            plt.close()

            plt.figure()
            plt.plot(generations, coop_rate, label='Cooperation Rate', color='orange', marker='o')
            plt.xlabel('Generation Number')
            plt.ylabel('Cooperation Rate')
            plt.title('Cooperation Rate over Generations')
            plt.legend()
            plt.ylim(0, 1)
            plt.savefig('plots/cooperation_rate.png')
            plt.close()

            # Overlayed plot with two y-axes
            fig, ax1 = plt.subplots()
            ax1.plot(generations, avg_hp, 'b-o', label='Average HP of Winners')
            ax1.set_xlabel('Generation Number')
            ax1.set_ylabel('Average HP of Winners', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax2 = ax1.twinx()
            ax2.plot(generations, coop_rate, 'r-o', label='Cooperation Rate')
            ax2.set_ylabel('Cooperation Rate', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 1)

            fig.suptitle('Average HP of Winners and Cooperation Rate over Generations')
            fig.tight_layout()
            fig.savefig('plots/hp_vs_cooperation.png')
            plt.close()

            plt.figure()
            plt.plot(generations, num_agents, label='Number of Agents', color='green', marker='o')
            plt.xlabel('Generation Number')
            plt.ylabel('Number of Agents')
            plt.title('Number of Agents over Generations')
            plt.legend()
            plt.savefig('plots/num_agents.png')
            plt.close()
        else:
            # More than one generation: update the existing plots with new data
            # Always plot the cumulative, continuous data
            plt.figure()
            plt.plot(generations, avg_hp, label='Average HP')
            plt.xlabel('Generation Number')
            plt.ylabel('Average HP')
            plt.title('Average HP over Generations')
            plt.legend()
            plt.savefig('plots/average_hp.png')
            plt.close()

            plt.figure()
            plt.plot(generations, coop_rate, label='Cooperation Rate', color='orange')
            plt.xlabel('Generation Number')
            plt.ylabel('Cooperation Rate')
            plt.title('Cooperation Rate over Generations')
            plt.legend()
            plt.ylim(0, 1)
            plt.savefig('plots/cooperation_rate.png')
            plt.close()

            # Overlayed plot with two y-axes
            fig, ax1 = plt.subplots()
            ax1.plot(generations, avg_hp, 'b-', label='Average HP of Winners')
            ax1.set_xlabel('Generation Number')
            ax1.set_ylabel('Average HP of Winners', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax2 = ax1.twinx()
            ax2.plot(generations, coop_rate, 'r-', label='Cooperation Rate')
            ax2.set_ylabel('Cooperation Rate', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 1)

            fig.suptitle('Average HP of Winners and Cooperation Rate over Generations')
            fig.tight_layout()
            fig.savefig('plots/hp_vs_cooperation.png')
            plt.close()

            plt.figure()
            plt.plot(generations, num_agents, label='Number of Agents', color='green')
            plt.xlabel('Generation Number')
            plt.ylabel('Number of Agents')
            plt.title('Number of Agents over Generations')
            plt.legend()
            plt.savefig('plots/num_agents.png')
            plt.close()

        # Cooperation distribution pie chart for the last generation
        alive_agents = [agent for agent in self.agents if agent.hp > 0]
        if alive_agents:
            coop_counts = [agent.last_action if agent.last_action is not None else 0 for agent in alive_agents]
            plt.figure()
            num_cooperate = sum(coop_counts)
            num_defect = len(coop_counts) - num_cooperate
            labels = ['Defect', 'Cooperate']
            sizes = [num_defect, num_cooperate]
            colors = ['lightcoral', 'lightskyblue']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title(f'Cooperation Distribution in Generation {self.generation}')
            plt.savefig('plots/coop_distribution.png')
            plt.close()

if __name__ == "__main__":
    MAX_GENERATIONS = 50
    sim = Simulation()
    round_num = 0
    while sim.generation < MAX_GENERATIONS:
        if sim.num_agents <= 1:
            break
        round_num += 1
        games_this_round = sim.run_round()
        if sim.num_agents > 0:
            sim.summary(round_num)
        else:
            # Save best agent so far even if all agents died
            if sim.agents:
                best_agent = max(sim.agents, key=lambda agent: agent.hp)
                if not os.path.exists('best_agents'):
                    os.makedirs('best_agents')
                np.savez('best_agents/best_agent.npz', w1=best_agent.w1, b1=best_agent.b1, w2=best_agent.w2, b2=best_agent.b2)
                print(f"Best agent saved with HP: {best_agent.hp:.2f}")
            print("Population extinct. Ending run.")
            sim.plot_history()
            break
    sim.plot_history()

    # Retrieve best agent
    if sim.agents:
        best_agent = max(sim.agents, key=lambda agent: agent.hp)
        if not os.path.exists('best_agents'):
            os.makedirs('best_agents')
        np.savez('best_agents/best_agent.npz', w1=best_agent.w1, b1=best_agent.b1, w2=best_agent.w2, b2=best_agent.b2)
        print(f"Best agent saved with HP: {best_agent.hp:.2f}")
