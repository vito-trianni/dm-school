import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx


num_agents = 50
num_options = 2
agents_state = [num_options]*num_agents
option_values = [10,10]
delta_speed = 0.1
k = 0.2
h = 0.8

T = 100


#G = nx.complete_graph(num_agents)
#G = nx.erdos_renyi_graph(num_agents, 0.1)
#G = nx.watts_strogatz_graph(num_agents, 5, 0.3)
#G = nx.barabasi_albert_graph(num_agents, 4)
G = nx.complete_graph(num_agents)

fig = plt.figure(figsize =(12,6))
ax1 = plt.subplot(121)
nx.draw(G, with_labels=True, font_weight='bold')

def get_neighbour( num_agents, agent, G = None):
    if G is not None:
        return np.random.choice(G[agent])

    n = agent
    while n == agent:
        n = np.random.randint(num_agents)
    return n


# iterate on agents to change their state:
stats = np.zeros((T,num_options+1))
for t in range(T):
    prev_state = agents_state.copy()
    for agent in range(num_agents):
        state = prev_state[agent]
        p = np.random.random()

        # the agent is uncommitted
        if state == num_options:
            # discovery
            current_option = np.random.randint(num_options)
            P_gamma = k*option_values[current_option]*delta_speed
            if p < P_gamma:
                agents_state[agent] = current_option
                continue

            # recruitment
            neighbour = get_neighbour(num_agents, agent, G)
            neighbour_option = prev_state[neighbour]
            if neighbour_option != num_options:
                P_rho = h*option_values[neighbour_option]*delta_speed
                if p < P_gamma+P_rho:
                    agents_state[agent] = neighbour_option
        else:
            P_alpha = k/option_values[state]*delta_speed
            neighbour = get_neighbour(num_agents, agent, G)
            P_sigma = 0
            if neighbour_option != num_options and neighbour_option != state:
                P_sigma = h*option_values[neighbour_option]*delta_speed
            if p < P_alpha + P_sigma:
                agents_state[agent] = num_options
    print( agents_state )


    # analysis
    for option in range(num_options+1):
        stats[t,option] = agents_state.count(option)

ax2 = plt.subplot(122)
ax2.plot(np.arange(T),stats[:,0], linewidth=4, label="option 0")
ax2.plot(np.arange(T),stats[:,1], linewidth=4, label="option 1")
ax2.plot(np.arange(T),stats[:,2], linewidth=4, label="uncommitted")
ax2.legend()

fig.savefig("dm.pdf")
plt.close()
