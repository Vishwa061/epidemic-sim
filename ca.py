import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

SEED = 109875089
random.seed(SEED)

class State:
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    VACCINATED = 3


class Node:
    COLOR_MAP = ['white', 'red', 'black', 'yellow']

    def __init__(self, row, col, state=State.SUSCEPTIBLE):
        self.name = f'r{row}c{col}'
        self.row = row
        self.col = col
        self.state = state
        self.duration = 0 # how long the node has been in the current state (in time steps)
        self.adoption_threshold = random.random()
        self.suseptiblity = random.random()
    
    def get_color(self):
        return Node.COLOR_MAP[self.state]


class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[Node(r, c) for c in range(cols)] for r in range(rows)]

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
    RECOVERY_TIME = 14

    def __init__(self, rows, cols, initial_infection_percent):
        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.edges = [] # directed edges: (from node, to node)
        self.waves = [] # list of waves where each wave is a list of edges
        self.SIR_over_time = [] # list of (t, (num of S nodes, num of I nodes, num of R nodes)) tuples
        self.time_steps = 0
        self.interval = Simulation.DEFAULT_INTERVAL
        self.frames = self.frames_gen
        self.ani = None
        self.vaccination_enabled = False
        self.num_nodes = rows * cols

        # infect a percentage of the nodes
        self.do_initial_infection(initial_infection_percent)

        # create graph
        self.G = nx.Graph(seed=SEED)
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

    def set_frames(self, frames):
        self.frames = frames

    def set_interval(self, interval):
        self.interval = interval

    def enable_vaccination(self):
        self.vaccination_enabled = True

    def disable_vaccination(self):
        self.vaccination_enabled = False

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
    
    def recover_nodes(self, nodes_to_recover):
        for n in nodes_to_recover:
            n.state = State.RECOVERED
            n.duration = 0
            self.G.nodes[n.name]['color'] = 'black'

    def vaccinate_nodes(self, nodes_to_vaccinate):
        for n in nodes_to_vaccinate:
            n.state = State.VACCINATED
            n.duration = 0
            self.G.nodes[n.name]['color'] = 'yellow'

    def increment_durations(self):
        for r in self.grid.grid:
            for n in r:
                n.duration += 1

    def get_num_susceptible_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.SUSCEPTIBLE or n.state == State.VACCINATED])

    def get_num_infected_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.INFECTED])

    def get_num_recovered_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.RECOVERED])

    def get_num_vaccinated_nodes(self):
        return len([n for n in self.grid.grid for n in n if n.state == State.VACCINATED])

    def step(self, current_time_step):
        nodes_to_infect = []
        nodes_to_recover = []
        nodes_to_vaccinate = []
        wave = []
        for r in self.grid.grid:
            for n in r:
                if n.state == State.INFECTED and n.duration >= Simulation.RECOVERY_TIME: # recover after 14 time steps
                    nodes_to_recover.append(n)
                elif n.state == State.SUSCEPTIBLE:
                    chance_of_infection = Simulation.CHANCE_OF_INFECTION * n.suseptiblity
                    if self.vaccination_enabled:
                        if n.state == State.SUSCEPTIBLE and n.adoption_threshold < (self.get_num_recovered_nodes() * 0.3 + self.get_num_vaccinated_nodes()) / self.num_nodes:
                            nodes_to_vaccinate.append(n)
                            continue
                        if n.state == State.VACCINATED:
                            chance_of_infection *= 0.1
                    neighbours = self.grid.get_neighbours(n.row, n.col)
                    for neighbour in neighbours:
                        if neighbour.state == State.INFECTED and random.random() < chance_of_infection:
                            nodes_to_infect.append(n)
                            wave.append((neighbour, n))
                            self.edges.append((neighbour, n))
                            break
        self.increment_durations()
        self.infect_nodes(nodes_to_infect)
        self.recover_nodes(nodes_to_recover)
        self.vaccinate_nodes(nodes_to_vaccinate)
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
                elif u.state == State.RECOVERED:
                    self.G.edges[u.name, v.name]['color'] = 'black'
                elif u.state == State.VACCINATED:
                    self.G.edges[u.name, v.name]['color'] = 'yellow'
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
        for n in self.G.nodes():
            node = self.grid.get_node(n)
            neighbours = self.grid.get_neighbours(node.row, node.col)
            for neighbour in neighbours:
                if neighbour.name != node.name and neighbour.state == State.INFECTED:
                    num_infectious_nodes += 1
                    break
        if num_infectious_nodes > 0:
            R0 = num_infected_edges / num_infectious_nodes
        R0_formated = str('{:10.4f}'.format(R0)).strip()

        # record SIR over time
        suseptible_nodes = self.get_num_susceptible_nodes()
        infected_nodes = self.get_num_infected_nodes()
        recovered_nodes = self.get_num_recovered_nodes()
        self.SIR_over_time.append((num, (suseptible_nodes, infected_nodes, recovered_nodes)))

        self.ax.clear()
        self.ax.set_title(f't = {str(num)} | R0 = {R0_formated} | S = {suseptible_nodes}, I = {infected_nodes}, R = {recovered_nodes}', fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        nx.draw(G=self.G, pos=self.pos, ax=self.ax, node_size=Simulation.NODE_SIZE, node_color=node_color_map, edge_color=edge_color_map)
        
        if num % 10 == 0:
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
        self.save_to_csv(self.edges, Simulation.EDGES_SAVE_FILENAME)

    def save_waves_to_csv(self):
        for i in range(len(self.waves)):
            self.save_to_csv(self.waves[i], f'{Simulation.WAVES_SAVE_FOLDER}/wave{i}.csv')

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
        plt.yscale('log')
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
    frames = 121
    interval = 1

    # run the simulation without vaccinations
    sim = Simulation(rows, cols, initial_infection_percent)
    sim.set_frames(frames)
    sim.set_interval(interval)
    sim.enable_vaccination()
    sim.start()

    # run analysis
    sim.analyze()
