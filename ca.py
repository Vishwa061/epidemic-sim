import random
from turtle import color
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class State:
    SUSCEPTIBLE = 0
    INFECTED = 1
    REMOVED = 2

class Node:
    def __init__(self, row, col, state=State.SUSCEPTIBLE):
        self.name = f'r{row}c{col}'
        self.row = row
        self.col = col
        self.state = state
        self.duration = 0 # how long the node has been in the current state (in time steps)

    def __str__(self):
        if self.state == State.SUSCEPTIBLE:
            return '.'
        elif self.state == State.INFECTED:
            return '#'
        elif self.state == State.REMOVED:
            return 'x'
        raise Exception(f'Invalid state: {self.state}')
    
    def get_color(self):
        if self.state == State.SUSCEPTIBLE:
            return 'white'
        elif self.state == State.INFECTED:
            return 'red'
        elif self.state == State.REMOVED:
            return 'black'
        raise Exception(f'Invalid state: {self.state}')


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[Node(r, c) for c in range(cols)] for r in range(rows)]

    def __str__(self):
        return '\n'.join([' '.join([str(n) for n in row]) for row in self.grid])

    def get_neighbours(self, row, col, radius=1):
        neighbours = []
        for r in range(max(0, row-radius), min(len(self.grid), row+radius+1)):
            for c in range(max(0, col-radius), min(len(self.grid[0]), col+radius+1)):
                if (r == row and c == col):
                    continue
                neighbours.append(self.grid[r][c])
        return neighbours
    
    def get_node(self, name):
        row = int(name[1:].split('c')[0])
        col = int(name[1:].split('c')[1])
        return self.grid[row][col]
      
class Simulation:
    WAVES_SAVE_FOLDER = 'ca_waves'
    EDGES_SAVE_FILENAME = 'ca_edges.csv'
    DEFAULT_INTERVAL = 200
    NODE_SIZE = 40
    CHANCE_OF_INFECTION = 0.1 # 10% chance of infection

    def __init__(self, rows, cols, initial_infection_percent, frames=None, interval=None, seed=None):
        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.edges = [] # directed edges: (from node, to node)
        self.waves = [] # list of waves where each wave is a list of edges
        self.removed = [] # list of removed nodes
        self.SIR_over_time = [] # list of (t, (num of S nodes, num of I nodes, num of R nodes)) tuples
        self.time_steps = 1
        self.interval = interval or self.DEFAULT_INTERVAL
        self.frames = frames or self.frames_gen
        self.seed = seed or random.randint(0, 1000000)
        self.ani = None

        # infect a percentage of the nodes
        self.do_initial_infection(initial_infection_percent)

        # create graph
        self.G = nx.Graph(seed=self.seed)
        self.fixed_positions = {}
        for row in self.grid.grid:
            for node in row:
                if node.state == State.INFECTED:
                    self.G.add_node(node.name, color='red')
                else:
                    self.G.add_node(node.name, color='green')
                for neighbour in self.grid.get_neighbours(node.row, node.col):
                    self.G.add_edge(node.name, neighbour.name, color='green')
                self.fixed_positions[node.name] = (node.col, abs(node.row-self.rows)) # populate dict to position nodes
        fixed_nodes = self.fixed_positions.keys()
        self.pos = nx.spring_layout(self.G, pos=self.fixed_positions, fixed=fixed_nodes)
        
        # create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def do_initial_infection(self, percent_to_infect):
        for r in self.grid.grid:
            for n in r:
                if random.random() < percent_to_infect:
                    n.state = State.INFECTED

    def infect_nodes(self, nodes_to_infect):
        for n in nodes_to_infect:
            n.state = State.INFECTED
            n.duration = 0
            self.G.nodes[n.name]['color'] = 'red'
    
    def remove_nodes(self, nodes_to_recover):
        for n in nodes_to_recover:
            n.state = State.REMOVED
            n.duration = 0
            self.G.nodes[n.name]['color'] = 'black'

    def increment_durations(self):
        for r in self.grid.grid:
            for n in r:
                n.duration += 1

    def get_num_susceptible_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.SUSCEPTIBLE])

    def get_num_infected_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.INFECTED])

    def get_num_removed_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.REMOVED])

    def step(self, current_time_step):
        nodes_to_infect = []
        nodes_to_remove = []
        wave = []
        for r in self.grid.grid:
            for n in r:
                if n.state == State.INFECTED and n.duration >= 5: # remove after 5 time steps 
                    nodes_to_remove.append(n)
                elif n.state == State.SUSCEPTIBLE:
                    neighbours = self.grid.get_neighbours(n.row, n.col)
                    for neighbour in neighbours:
                        if neighbour.state == State.INFECTED and random.random() < self.CHANCE_OF_INFECTION:
                            nodes_to_infect.append(n)
                            wave.append((neighbour, n))
                            self.edges.append((neighbour, n))
                            break
        self.increment_durations()
        self.infect_nodes(nodes_to_infect)
        self.remove_nodes(nodes_to_remove)
        self.waves.append(wave)
        self.time_steps += 1
        self.display_graph(current_time_step)

    def display_graph(self, num):
        for e in self.G.edges.data():
            u = self.grid.get_node(e[0])
            v = self.grid.get_node(e[1])
            if u.state == v.state:
                if u.state == State.INFECTED:
                    self.G.edges[u.name, v.name]['color'] = 'red'
                elif u.state == State.REMOVED:
                    self.G.edges[u.name, v.name]['color'] = 'black'
                else:
                    self.G.edges[u.name, v.name]['color'] = 'green'

        node_color_map = [c[1]['color'] for c in self.G.nodes.data()]
        edge_color_map = [self.G[u][v]['color'] for u,v in self.G.edges]

        # calculate R0
        R0 = 0
        num_infected_edges = 0
        num_infectious_nodes = 0
        for u,v in self.G.edges:
            if self.G[u][v]['color'] == 'red':
                num_infected_edges += 1
                node = self.grid.get_node(v)
                neighbours = self.grid.get_neighbours(node.row, node.col)
                for neighbour in neighbours:
                    if neighbour.name != node.name and neighbour.state == State.INFECTED:
                        num_infectious_nodes += 1
                        break
        if num_infectious_nodes > 0:
            R0 = num_infected_edges / num_infectious_nodes
        R0_formated = str('{:10.4f}'.format(R0)).strip()

        suseptible_nodes = 0
        infected_nodes = 0
        removed_nodes = 0
        for c in node_color_map:
            if c == 'red':
                infected_nodes += 1
            elif c == 'green':
                suseptible_nodes += 1
            else:
                removed_nodes += 1
        self.SIR_over_time.append((num, (suseptible_nodes, infected_nodes, removed_nodes)))

        self.ax.clear()
        self.ax.set_title(f't = {str(num)} | R0 = {R0_formated} | S = {suseptible_nodes}, I = {infected_nodes}, R = {removed_nodes}', fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        nx.draw(G=self.G, pos=self.pos, ax=self.ax, node_size=self.NODE_SIZE, node_color=node_color_map, edge_color=edge_color_map)
        
        if num % 29 == 0:
            plt.savefig(f'ca_propagation/ca_propagation{str(num)}.png', bbox_inches='tight')

    def frames_gen(self):
        while self.get_num_infected_nodes() > 0:
            yield self.time_steps

    def start(self):
        self.display_graph(0)
        self.ani = animation.FuncAnimation(self.fig, self.step, frames=self.frames, interval=self.interval, repeat=False)
        plt.show()

    def save_to_csv(self, list_of_tuples, filename):
        with open(filename, 'w') as f:
            for t in list_of_tuples:
                line = ''
                for i in range(len(t)):
                    line += str(t[i]) + ','
                line = line[:-1] + '\n'
                f.write(line)
    
    def save_edges_to_csv(self):
        self.save_to_csv(self.edges, self.EDGES_SAVE_FILENAME)

    def save_waves_to_csv(self):
        for i in range(len(self.waves)):
            self.save_to_csv(self.waves[i], f'{self.WAVES_SAVE_FOLDER}/wave{i}.csv')

    def analyze(self):
        # create infection graph
        G = nx.DiGraph(self.edges)

        # histogram for out-degree distribution
        out_degree_list = [d[1] for d in G.out_degree()]
        plt.bar(*np.unique(out_degree_list, return_counts=True))
        plt.title('Out-Degree Distribution')
        plt.xlabel('Out-Degree')
        plt.ylabel('Number of Nodes')
        plt.savefig(f'ca_analysis/out_degree_dist.png', bbox_inches='tight')
        plt.clf()

        list_of_num_susceptible_nodes = [t[1][0] for t in self.SIR_over_time]
        list_of_num_infected_nodes = [t[1][1] for t in self.SIR_over_time]
        list_of_num_removed_nodes = [t[1][2] for t in self.SIR_over_time]
        list_of_time_steps = [t[0] for t in self.SIR_over_time]

        # plot SIR over time
        plt.plot(list_of_time_steps, list_of_num_susceptible_nodes, label='S', color='green')
        plt.plot(list_of_time_steps, list_of_num_infected_nodes, label='I', color='red')
        plt.plot(list_of_time_steps, list_of_num_removed_nodes, label='R', color='black')
        plt.title('SIR Over Time')
        plt.xlabel('Time steps')
        plt.ylabel('Number of Nodes')
        plt.legend(['Susceptible (S)', 'Infected (I)', 'Removed (R)'], loc='center right')
        plt.savefig(f'ca_analysis/SIR_over_time.png', bbox_inches='tight')
        plt.clf()
        
        # TODO: x-axis is diameter of the graph, y-axis is number of nodes whose out-degree is greater or equal to some threshold
        # threshold_list = np.unique(out_degree_list)
        # D = nx.DiGraph(self.edges)
        # plt.plot(G)
        # plt.show()
        # for threshold in threshold_list:
        #     nodes_to_remove = [n for n in D.nodes if D.out_degree(n) < threshold]
        #     D.remove_nodes_from(nodes_to_remove)
        #     print(f'{threshold} nodesss : {len(D.nodes)}')
            # find diameter of the graph
            # diameter = nx.diameter(D)
            # print(f'Diameter of the graph with threshold {threshold} is {diameter}')

if __name__ == '__main__':
    # sim parameters
    rows = 40
    cols = 40
    initial_infection_percent = 0.01
    seed = 4561215534
    frames = 100
    interval = 200

    # seed the random number generator
    random.seed(seed)

    # create and run the simulation
    sim = Simulation(rows, cols, initial_infection_percent, interval=1, seed=seed)
    sim.start()

    # run analysis
    sim.analyze()
