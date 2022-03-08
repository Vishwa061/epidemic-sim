import random
import networkx as nx
import matplotlib.pyplot as plt

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
    def __init__(self, rows, cols, initial_infection_percent):
        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.edges = [] # directed edges: (from node, to node)
        self.time_steps = 0
        self.do_initial_infection(initial_infection_percent)

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

    def step(self):
        nodes_to_infect = []
        nodes_to_remove = []
        for r in self.grid.grid:
            for n in r:
                if n.state == State.INFECTED:
                    if n.duration >= 5: # remove after 5 time steps
                        nodes_to_remove.append(n)
                elif n.state == State.SUSCEPTIBLE:
                    neighbours = self.grid.get_neighbours(n.row, n.col)
                    for neighbour in neighbours:
                        if neighbour.state == State.INFECTED and random.random() < 0.08: # 100% chance of infection
                            nodes_to_infect.append(n)
                            self.edges.append((neighbour.name, n.name))
                            break
        self.increment_durations()
        self.infect_nodes(nodes_to_infect)
        self.remove_nodes(nodes_to_remove)
        self.time_steps += 1

    def run(self, time_steps):
        print(self.grid)
        for _ in range(time_steps):
            self.step()
            print(f'\n{self.grid}')

    def full_run(self):
        print(self.grid)
        while self.get_num_infected_nodes() > 0:
            self.step()
            print(f'\n{self.grid}')

    def save_to_csv(self, list_of_tuples, filename):
        with open(filename, 'w') as f:
            for t in list_of_tuples:
                line = ''
                for i in range(len(t)):
                    line += str(t[i]) + ','
                line = line[:-1] + '\n'
                f.write(line)
    
    def save_edges_to_csv(self, filename):
        self.save_to_csv(self.edges, filename)

if __name__ == '__main__':
    # sim parameters
    rows = 20
    cols = 20
    initial_infection_percent = 0.01
    time_steps = 2
    seed = 8486

    # seed the random number generator
    random.seed(seed)

    # create and run the simulation
    simulation = Simulation(rows, cols, initial_infection_percent)
    simulation.run(time_steps)
    # simulation.full_run()

    # save the edges to a csv file
    simulation.save_edges_to_csv('edges.csv')

    # # plot directed graph
    # G = nx.DiGraph()
    # G.add_edges_from(simulation.edges)
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_size=100)
    # nx.draw_networkx_edges(G, pos, width=1)
    # plt.axis('off')
    # plt.show()
