import matplotlib
matplotlib.use('TKAgg')

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

plt.style.use('dark_background')

def nearest_neighbours(row, col, latticeLen):
    return [((row - 1) % latticeLen, (col - 1) % latticeLen), # One up, one left
            ((row - 1) % latticeLen, col),                    # One up
            ((row - 1) % latticeLen, (col + 1) % latticeLen), # One up, one right
            (row, (col - 1) % latticeLen),                    # One left
            (row, (col + 1) % latticeLen),                    # One right
            ((row + 1) % latticeLen, (col - 1) % latticeLen), # One down, one left
            ((row + 1) % latticeLen, col),                    # One down
            ((row + 1) % latticeLen, (col + 1) % latticeLen)] # One down, one right

class GameOfLife():

    def __init__(self, n, initialization='random'):
        self.n = n

        if initialization == 'random':
            self.states = np.random.randint(0, 2, size=(n, n))
        elif initialization == 'oscillator':
            self.states = np.zeros((n, n))
            self.states[np.array([25, 25, 25]), np.array([24, 25, 26])] = 1
        elif initialization == 'glider':
            self.states = np.zeros((n, n))
            self.states[np.array([0, 1, 2, 2, 2]), np.array([1, 2, 0, 1, 2])] = 1
        else:
            raise Exception("Unknown initial configuration.")

    def update_lattice(self):
        newStates = np.copy(self.states)
        for i in range(self.n):
            for j in range(self.n):
                aliveNeigh = 0
                for neighbour in nearest_neighbours(i, j, self.n):
                    aliveNeigh += self.states[neighbour]
                if self.states[i, j] == 1:
                    if aliveNeigh < 2 or aliveNeigh > 3:
                        newStates[i, j] = 0
                elif self.states[i, j] == 0:
                    if aliveNeigh == 3:
                        newStates[i, j] = 1
        self.states = newStates

    def animate(self, nstep):
        fig = plt.figure(figsize=(8, 8))
        #fig.patch.set_facecolor('blue')
        #ig.patch.set_alpha(0.4)
        plt.tight_layout()
        im=plt.imshow(self.states, animated=True)

        for n in range(nstep):
            self.update_lattice()

            plt.cla()
            plt.tight_layout()
            im=plt.imshow(self.states, animated=True)
            plt.draw()
            plt.pause(0.01)

    def run_until_equilibrium(self):
        activeSites = np.sum(self.states)
        timesteps = 0

        self.update_lattice()
        activeSites1 = np.sum(self.states)
        timesteps += 1

        self.update_lattice()
        activeSites2 = np.sum(self.states)
        timesteps += 1

        while activeSites != activeSites1 or activeSites1 != activeSites2:
            self.update_lattice()
            activeSites = activeSites1
            activeSites1 = activeSites2
            activeSites2 = np.sum(self.states)
            timesteps += 1
            if timesteps > 3000:
                return timesteps
        return timesteps

    def get_centre_of_mass(self):
        centre = np.array([0, 0])
        pointsNum = 0
        for i in range(self.n):
            for j in range(self.n):
                if self.states[i, j] == 1:
                    if i == self.n - 1 or i == 0 or j == self.n - 1 or j == 0:
                        return None
                    else:
                        centre += np.array([i, j])
                        pointsNum += 1
        return centre / pointsNum

def task2():
    times = []

    f = open('equilibrate_times.dat', 'w')
    for i in range(100):
        time = GameOfLife(50, 'random').run_until_equilibrium()
        times.append(time)
        f.write(f'{time}\n')
    f.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(times, bins=50)
    ax.set_xlabel('Number of iterations')
    ax.set_title('Time taken to reach equilibrium for 100 random initializations of Game of Life')
    plt.savefig('task_2')
    plt.show()
    plt.close()

def task3():
    # Get centre of mass for 50 iterations
    gol = GameOfLife(50, initialization='glider')
    centres = []
    for i in range(200):
        centres.append(gol.get_centre_of_mass())
        gol.update_lattice()
    newCentres = []
    times = []
    for i in range(len(centres)):
        if centres[i] is not None:
            newCentres.append(centres[i])
            times.append(i)

    speeds = [(np.linalg.norm(newCentres[x]) - np.linalg.norm(newCentres[0]))/(times[x] - times[0]) for x in range(1, len(newCentres))]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(times[1:], speeds)
    ax.set_xlabel('Time from initial measurement.')
    ax.set_ylabel('Average velocity measured from the initial measurement.')
    plt.savefig('task_3_figure')
    plt.show()
    plt.close()

#task2()


# GameOfLife(10).run_until_equilibrium()

# input

# if(len(sys.argv) != 3):
#    print("Usage python game_of_life.py n mode")

# n=int(sys.argv[1])
# mode=str(sys.argv[2])

n = 50
mode = 'random'
nupdates = 100

gol = GameOfLife(n, mode)
gol.animate(nupdates)