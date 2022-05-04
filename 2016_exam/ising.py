import matplotlib
matplotlib.use('TKAgg')

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from auxiliary_functions import *

def glauber_energy_change(spins, spinFlipLoc, j, h):
    # Assume the energy constant J, magfield h
    # Final delta E is 2J(S_t + S_l + S_r + S_b)*S_m +2hS_m
    rowloc, colloc = spinFlipLoc

    lrows, lcols = spins.shape

    # Find the nearest neighbours: [(upper), (left), (right), (lower)] (periodic boundary conditions apply)
    nearestNeigbours = nearest_neighbours(rowloc, colloc, lrows)

    oldSpin = spins[rowloc, colloc]
    newSpin = -oldSpin
    neighbourSpinSum = 0
    for neigbour in nearestNeigbours:
        row, col = neigbour
        neighbourSpinSum += spins[row, col]

    return 2*j*neighbourSpinSum*oldSpin + 2*h*oldSpin

def metropolis_test_true(deltaE, temp):
    if deltaE <= 0:
        return True
    else:
        prob = np.exp(-deltaE / temp)
        if prob >= np.random.random():
            return True
        else:
            return False

def sng(row, col):
    return (-1)**(row+col)

def sweep(lat, j, h, kt, l):
    for i in range(l):
        for k in range(l):
            itrial=np.random.randint(0,l)
            jtrial=np.random.randint(0,l)
            deltaE = glauber_energy_change(lat, (itrial, jtrial), j, h)
            spinnew = -lat[itrial,jtrial]
            if metropolis_test_true(deltaE, kt):
                lat[itrial,jtrial] = spinnew
    return lat

def init_state(l):
    # random initial state
    lat=np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            r=random.random()
            if(r<0.5): lat[i,j]=-1
            if(r>=0.5): lat[i,j]=1
    return lat

def magnetisation(lat):
    return np.sum(lat)

def stagmag(lat):
    L = len(lat)
    M = 0
    for i in range(L):
        for j in range(L):
            M += lat[i,j]*sng(i,j)
    return M

def energy(lat, j, h):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    N = len(lat)
    
    # neighbour sums
    for i in range(N):
        for j in range(N):
            S = lat[i,j]
            nb = lat[(i+1)%N, j] + lat[i,(j+1)%N] + lat[(i-1)%N, j] + lat[i,(j-1)%N]
            energy += -j*nb*S
    energy = energy/2 # compensate for overcounting
    # magnetic field contribution
    for i in range(N):
        for j in range(N):
            energy += -h*lat[i,j]
    return energy

# Define values used throughout
L = 50
J = -1
KT = 1

#if len(sys.argv) != 2:
#    raise Exception('Usage python ising.py h')

#h = float(sys.argv[1])

hs = np.linspace(0, 10, 21)
measures = [[],[],[]] # They will be in order: magnetisation, stagered magnetisation, energy
vars = [[],[],[]]

for h in tqdm(hs):
    lattice = init_state(L)
    results = simulate_and_measure(lattice, 1000, 
                                    sweep, magnetisation, stagmag, energy,
                                    equilibration=100, upint=10, std='std',
                                    sweep=(J, h, KT, L), magnetisation=(),
                                    stagmag=(), energy=(J, h))
    for i, result in enumerate(results):
        measures[i].append(result[0])
        vars[i].append(result[1])

paths = ['2016_exam/magnetisation_h', '2016_exam/magstag_h', '2016_exam/energy_h']
xlabels = ['Value of h'] * 3
ylabels = ['Average magnetisation', 'Average staggered magnetisation', 'Average energy']

for i in range(len(measures)):
    scatter_and_save_data(hs, measures[i], paths[i], xlabels[i], ylabels[i], '', vars[i])
