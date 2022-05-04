import numpy as np
from auxiliary_functions import *

L = 50

def nearest_neighbours(row, col, latticeLen):
    return [((row - 1) % latticeLen, col), 
            (row, (col - 1) % latticeLen), 
            (row, (col + 1) % latticeLen), 
            ((row + 1) % latticeLen, col)]

def initial_state():
    lattice = np.zeros(shape=(L,L))
    for i in range(L):
        for j in range(L):
            lattice[i,j] = np.random.randint(0, 2)
    return lattice

#@jit(nopython=True)
def sweep(lat, p):
    for i in range(L):
        for j in range(L):
            itrial = np.random.randint(0,L)
            jtrial = np.random.randint(0,L)
            if lat[itrial, jtrial] == 1: # Something only happens if the selected cell is active
                if np.random.random() < 1-p:
                    lat[itrial, jtrial] = 0
                else: # np.random.random() < p:
                    nearNeigh = nearest_neighbours(itrial, jtrial, L)
                    lat[nearNeigh[np.random.randint(0, 4)]] = 1
    return lat

lattice = initial_state()
animate(lattice, 300, sweep, 1, 0.7)