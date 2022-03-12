import random
from timeit import repeat
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

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
        else:
            return '?'

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
      
class Simulation:
    WAVES_SAVE_FOLDER = 'waves'
    EDGES_SAVE_FILENAME = 'edges.csv'
    DEFAULT_INTERVAL = 200

    def __init__(self, rows, cols, initial_infection_percent):
        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.edges = [] # directed edges: (from node, to node)
        self.waves = [] # list of waves (each wave is a list of edges)
        self.time_steps = 0
        self.do_initial_infection(initial_infection_percent)

        # create graph 
        self.G = nx.Graph()
        self.fixed_positions = {}
        for row in self.grid.grid:
            for node in row:
                for neighbour in self.grid.get_neighbours(node.row, node.col):
                    self.G.add_edge(node.name, neighbour.name, color='green')
                self.fixed_positions[node.name] = (node.col, abs(node.row-self.rows)) # populate dict to position nodes

        fixed_nodes = self.fixed_positions.keys()
        self.pos = nx.spring_layout(self.G, pos=self.fixed_positions, fixed=fixed_nodes)
        
        # create figure
        self.fig, self.ax = plt.subplots()

    def do_initial_infection(self, percent_to_infect):
        for r in self.grid.grid:
            for n in r:
                if random.random() < percent_to_infect:
                    n.state = State.INFECTED

    def infect_nodes(self, nodes_to_infect):
        for n in nodes_to_infect:
            n.state = State.INFECTED
            n.duration = 0
    
    def remove_nodes(self, nodes_to_recover):
        for n in nodes_to_recover:
            n.state = State.REMOVED
            n.duration = 0

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

    def step(self, num):
        nodes_to_infect = []
        nodes_to_remove = []
        wave = []
        for r in self.grid.grid:
            for n in r:
                if n.state == State.INFECTED:
                    if n.duration >= 5: # remove after 5 time steps
                        nodes_to_remove.append(n)
                    else:
                        neighbours = self.grid.get_neighbours(n.row, n.col)
                        for neighbour in neighbours:
                            if neighbour.state == State.SUSCEPTIBLE and random.random() < 0.116: # 11.6% chance of infection:
                                nodes_to_infect.append(neighbour)
                                wave.append((n, neighbour))
                                self.edges.append((n, neighbour))
                                # if n.row == 0  and n.col == 22:
                                #     print(f'{n.name} was infected by {neighbour.name}')
        self.increment_durations()
        self.infect_nodes(nodes_to_infect)
        self.remove_nodes(nodes_to_remove)
        self.waves.append(wave)
        self.time_steps += 1
        self.display_graph(num)

    def display_graph(self, num):
        node_color_map = []
        edge_color_map = []
        for row in self.grid.grid:
            for node in row:
                if node.state == State.INFECTED:
                    node_color_map.append('red')
                elif node.state == State.REMOVED:
                    node_color_map.append('black')
                else:
                    node_color_map.append('green')
                
                if node.row == 1 and node.col == 22 and node.state == State.INFECTED:
                    node_color_map.pop()
                    node_color_map.append('blue')
                    print(f'{node.name} is blue')
                    print(f'{node.name} is {node.state}')
                
                neighbours = self.grid.get_neighbours(node.row, node.col)
                for neighbour in neighbours:
                    if self.G.edges[neighbour.name, node.name] and node.state == neighbour.state:
                        edge = self.G.edges[neighbour.name, node.name]
                        if node.state == State.INFECTED:
                            edge['color'] = 'red'
                        elif node.state == State.REMOVED:
                            edge['color'] = 'black'
                        else:
                            edge['color'] = 'green'
                            
        edge_color_map = [self.G[u][v]['color'] for u,v in self.G.edges]
        self.ax.clear()
        nx.draw(G=self.G, pos=self.pos, ax=self.ax, node_size=40, node_color=node_color_map, edge_color=edge_color_map) # with_labels=True, font_size=8
        
        self.ax.set_title(f't = {str(num)}', fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def frames(self):
        while self.get_num_infected_nodes() > 0:
            yield self.time_steps

    def start(self, time_steps=None, interval=None):
        interval = interval or self.DEFAULT_INTERVAL
        frames = time_steps or self.frames
        ani = animation.FuncAnimation(self.fig, self.step, frames=frames, interval=interval, repeat=False)
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

if __name__ == '__main__':
    # sim parameters
    rows = 30
    cols = 30
    initial_infection_percent = 0.01
    seed = 8486

    # seed the random number generator
    random.seed(seed)

    # create and run the simulation
    sim = Simulation(rows, cols, initial_infection_percent)
    sim.start(time_steps=11, interval=1)
    # for node in sim.grid.grid[0]:
    #     if node.state == State.INFECTED:
    #         print("NODE:"+node.name)
    #         nn = sim.grid.get_neighbours(node.row, node.col)
    #         for nnn in nn:
    #             print(nnn.name)

    # save the edges to a csv file
    # sim.save_edges_to_csv()

    # save the waves to a csv file
    sim.save_waves_to_csv()

    # create a grid to visualize the simulation
    # G = nx.DiGraph()
    # fixed_positions = {}
    # for row in sim.grid.grid:
    #     for node in row:
    #         G.add_node(node.name)
    #         fixed_positions[node.name] = (node.row, node.col) # populate dict to position nodes

    # for wave in sim.waves:
    #     G.add_edges_from(wave)
    #     fixed_nodes = fixed_positions.keys()
    #     pos = nx.spring_layout(G, pos=fixed_positions, fixed=fixed_nodes)

    #     color_map = []
    #     for row in sim.grid.grid:
    #         for node in row:
    #             color_map.append('green')
    #     nx.draw(G, pos, node_size=40, node_color=color_map)
    #     plt.show()

    #----------------------------------------------------------------------------------------------------------------------


    #     nx.draw_networkx_nodes(G, pos, node_size=100)
    #     nx.draw_networkx_edges(G, pos, width=1)
    #     plt.axis('off')
    #     plt.show()

    # # plot directed graph
    # G = nx.DiGraph()
    # G.add_edges_from(sim.edges)

    # # get gcc
    # gcc_set = max(nx.weakly_connected_components(G), key=len)
    # print(nx.subgraph(G, gcc_set))

    # # get avg node degree
    # out_node_degrees = G.number_of_edges() / G.number_of_nodes()
    # print(out_node_degrees)

    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_size=100)
    # nx.draw_networkx_edges(G, pos, width=1)
    # plt.axis('off')
    # plt.show()
