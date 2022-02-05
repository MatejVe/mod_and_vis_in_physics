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
def glauber_energy_change(spins, spinFlipLoc):
    # Assume the energy constant J = 1
    # Final delta E is 2J(S_t + S_l + S_r + S_b)*S_m
    rowloc, colloc = spinFlipLoc

    lrows, lcols = spins.shape

    # Find the nearest neighbours: [(upper), (left), (right), (lower)] (periodic boundary conditions apply)
    nearestNeigbours = [((rowloc - 1) % lrows, colloc), 
                        (rowloc, (colloc - 1) % lcols),
                        (rowloc, (colloc + 1) % lcols),
                        ((rowloc + 1) % lrows, colloc)]

    oldSpin = spins[rowloc, colloc]
    newSpin = -oldSpin
    neighbourSpinSum = 0
    for neigbour in nearestNeigbours:
        row, col = neigbour
        neighbourSpinSum += spins[row, col]

    return 2*neighbourSpinSum*oldSpin  # TODO: check if this is always true

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
def glauber_ising_sim(nstep, N, kT, spin):
    print(f'Current temperature is {kT}.')
    J=1.0
    # nstep=10000

    mags = [] # Store the measured magnetisations
    mags2 = []
    energies = []
    energies2 = []

    #update loop here - for Glauber dynamics
    for n in range(nstep):
        if n % 1000 == 0:
            print(f'{n//100}% done.')
        for i in range(N):
            for j in range(N):

                #select spin randomly
                itrial=np.random.randint(0,N)
                jtrial=np.random.randint(0,N)
                spin_new=-spin[itrial,jtrial]

                #compute delta E eg via function (account for periodic BC)
                deltaE = glauber_energy_change(spin, (itrial, jtrial))

                #perform metropolis test
                if metropolis_test_true(deltaE, kT):
                    spin[itrial, jtrial] = spin_new

        if n > 1000:
            if n % 10 == 0:
                mag = np.sum(spin)  # total magnetisation
                mags.append(mag)
                mags2.append(np.power(mag, 2))
                energy = calcEnergy(spin)
                energies.append(energy)
                energies2.append(np.power(energy, 2))

    mags = np.array(mags)
    energies = np.array(energies)
    return np.mean(mags), np.mean(mags2), np.mean(energies), np.mean(energies2), spin

temps = np.linspace(1, 3, 21)
magAvgs = []
magAvgs2 = []
energyAvgs = []
energyAvgs2 = []

spin = np.ones((50, 50))

f = open('task4_results', 'w')
for temp in temps:
    magAvg, magAvg2, energyAvg, energyAvg2, spin = glauber_ising_sim(11000, 50, temp, spin)
    f.write(f'{temp} {magAvg} {magAvg2} {energyAvg} {energyAvg2}\n')

    magAvgs.append(magAvg)
    magAvgs2.append(magAvg2)
    energyAvgs.append(energyAvg)
    energyAvgs2.append(energyAvg2)
f.close()


fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes[0, 0].scatter(temps, magAvgs, marker='o', color='IndianRed')
axes[0, 0].set_xlabel('Value of kT')
axes[0, 0].set_ylabel('Total magnetization')

suss = []
for temp, magAvg, magAvg2 in zip(temps, magAvgs, magAvgs2):
    val = 1 / (2500 * temp) * (magAvg2 - magAvg**2)
    suss.append(val)

axes[0, 1].scatter(temps, suss, marker='o', color='RoyalBlue')
axes[0, 1].set_xlabel('Value of kT')
axes[0, 1].set_ylabel('Value of susceptibility')

axes[1, 0].scatter(temps, energyAvgs, marker='o', color='IndianRed')
axes[1, 0].set_xlabel('Value of kT')
axes[1, 0].set_ylabel('Total energy')

heatCapacities = []
for temp, eAvg, eAvg2 in zip(temps, energyAvgs, energyAvgs2):
    val = 1 / (2500 * temp**2) * (eAvg2 - eAvg**2)
    heatCapacities.append(val)

axes[1, 1].scatter(temps, heatCapacities, marker='o', color='RoyalBlue')
axes[1, 1].set_xlabel('Value of kT')
axes[1, 1].set_ylabel('Heat capacity per spin')

plt.tight_layout()
plt.savefig('Task4_fig')
plt.close()