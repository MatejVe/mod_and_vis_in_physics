from hashlib import new
import matplotlib
from pyparsing import col
matplotlib.use('TKAgg')

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

from numba import jit, cuda
@jit
def are_neighbours(row1, col1, row2, col2, latticeLen):
    if abs(row2 - row1) + abs(col2 - col1) == 1:
        return True
    if abs(row2 - row1) + abs(col2 - col1) == latticeLen - 1:
        if abs(row2 - row1) == 0 or abs(col2 - col1) == 0:
            return True
    return False

@jit
def nearest_neighbours(row, col, latticeLen):
    return [((row - 1) % latticeLen, col), 
            (row, (col - 1) % latticeLen), 
            (row, (col + 1) % latticeLen), 
            ((row + 1) % latticeLen, col)]

@jit
def kawasaki_energy_change(spins, spinFlipLocs):
    # Assume the energy constant J = 1 and that the two spins in question have different orientations (+1 and -1)
    row1, col1, row2, col2 = spinFlipLocs
    spin1, spin2 = spins[row1, col1], spins[row2, col2]

    lrows, lcols = spins.shape

    deltaE = 0
    if are_neighbours(row1, col1, row2, col2, lrows):
        for neighbour in nearest_neighbours(row1, col1, lrows):
            row, col = neighbour
            if row != row2 and col != col2:
                deltaE += spins[row2, col2] * spins[row, col]
        for neighbour in nearest_neighbours(row2, col2, lrows):
            row, col = neighbour
            if row != row1 and col != col2:
                deltaE += spins[row1, col1] * spins[row, col]
    else:
        for neighbour in nearest_neighbours(row1, col1, lrows):
            row, col = neighbour
            deltaE += spins[row2, col2] * spins[row, col]
        for neighbour in nearest_neighbours(row2, col2, lrows):
            row, col = neighbour
            deltaE += spins[row1, col1] * spins[row, col]
    return -2*deltaE

@jit
def metropolis_test_true(deltaE, temp):
    if deltaE <= 0:
        return True
    else:
        prob = np.exp(-deltaE / temp)
        if prob >= np.random.random():
            return True
        else:
            return False

@jit
def calcEnergy(spins):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    N = len(spins)
    
    for i in range(len(spins)):
        for j in range(len(spins)):
            S = spins[i,j]
            nb = spins[(i+1)%N, j] + spins[i,(j+1)%N] + spins[(i-1)%N, j] + spins[i,(j-1)%N]
            energy += -nb*S
    return energy/2.  # to compensate for over-counting

@jit
def kawasaki_ising_sim(nstep, N, kT):
    print(f'Current temperature is {kT}.')
    J=1.0
    # nstep=10000

    spin=np.zeros((N,N),dtype=float)

    #initialise spins randomly

    for i in range(N):
        for j in range(N):
            r=random.random()
            if(r<0.5): spin[i,j]=-1
            if(r>=0.5): spin[i,j]=1

    energies = [] # Store the measured magnetisations
    energies2 = []

    #update loop here - for Glauber dynamics
    for n in range(nstep):
        if n % 1000 == 0:
            print(f'{n//100}% done.')
        for i in range(N):
            for j in range(N):

                #select spins randomly
                i1=np.random.randint(0,N)
                j1=np.random.randint(0,N)
                spin1 = spin[i1, j1]
                i2=np.random.randint(0,N)
                j2=np.random.randint(0,N)
                spin2 = spin[i2, j2]

                #compute delta E eg via function (account for periodic BC)
                deltaE = kawasaki_energy_change(spin, (i1, j1, i2, j2))

                #perform metropolis test
                if metropolis_test_true(deltaE, kT):
                    spin[i1, j1] = spin2
                    spin[i2, j2] = spin1

        if n > 1000:
            if n % 10 == 0:
                energy = calcEnergy(spin)
                energies.append(energy)
                energies2.append(np.power(energy, 2))

    energies = np.array(energies)
    return np.mean(energies), np.mean(energies2)

temps = np.linspace(1, 3, 21)
energyAvgs = []
energyAvgs2 = []

f = open('task5_results', 'w')
for temp in temps:
    e, e2 = kawasaki_ising_sim(11000, 50, temp)
    f.write(f'{temp} {e} {e2}\n')

    energyAvgs.append(e)
    energyAvgs2.append(e2)
f.close()

fig, axes = plt.subplots(2, 1, figsize=(18, 10))
axes[0].scatter(temps, energyAvgs, marker='o', color='IndianRed')
axes[0].set_xlabel('Value of kT')
axes[0].set_ylabel('Total energy')

heatCapacities = []
for temp, eAvg, eAvg2 in zip(temps, energyAvgs, energyAvgs2):
    val = 1 / (2500 * temp**2) * (eAvg2 - eAvg**2)
    heatCapacities.append(val)

axes[1].scatter(temps, heatCapacities, marker='o', color='RoyalBlue')
axes[1].set_xlabel('Value of kT')
axes[1].set_ylabel('Heat capacity per spin')

plt.savefig('Task5_fig')
plt.close()