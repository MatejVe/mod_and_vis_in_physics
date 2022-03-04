from email.errors import MessageError
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

    meanEnergy = np.mean(energies)
    meanEnergySquared = np.mean(energies2)
    heatCapacity = 1 / (N**2 * kT**2) * (meanEnergySquared - meanEnergy**2)

    meanMag = np.mean(mags)
    meanMagSquared = np.mean(mags2)
    suss = 1 / (N**2 * kT) * (meanMagSquared - meanMag**2)

    # Jackknife errors
    energyError = []
    heatCapacityError = []
    magError = []
    sussError = []

    for i in range(len(energies)):
        newEnergies = energies[:i] + energies[i+1:]
        newEnergiesSquared = [energy**2 for energy in newEnergies]

        newEmean = np.mean(newEnergies)
        meanSquared = np.mean(newEnergiesSquared)

        newHeat = 1 / (2500 * kT**2) * (meanSquared - newEmean**2)
        energyError.append((newEmean - meanEnergy)**2)
        heatCapacityError.append((heatCapacity - newHeat)**2)

        newMags = mags[:i] + mags[i+1:]
        newMagsSquared = [mag**2 for mag in newMags]

        newMagMean = np.mean(newMags)
        newMagMeanSquared = np.mean(newMagsSquared)

        newSuss = 1 / (N**2 * kT) * (newMagMeanSquared - newMagMean**2)
        magError.append((newMagMean - meanMag)**2)
        sussError.append((newSuss - suss)**2)
    
    energyError = np.sqrt(np.sum(energyError))
    heatCapacityError = np.sqrt(np.sum(heatCapacityError))
    magError = np.sqrt(np.sum(magError))
    sussError = np.sqrt(np.sum(sussError))

    toReturn = (meanEnergy, energyError,
                heatCapacity, heatCapacityError,
                meanMag, magError,
                suss, sussError,
                spin)
    return  toReturn

temps = np.linspace(1, 3, 21)
energyAvgs = []
energyErrors = []
heatCaps = []
hCerrors = []
magAvgs = []
magErrors = []
susses = []
sussErrors = []

spin = np.ones((50, 50))

f = open('task4_witherror_results', 'w')
for temp in temps:
    e, eerror, hC, hCerror, mag, magError, suss, sussError, spin = glauber_ising_sim(11000, 50, temp, spin)
    f.write(f'{temp} {e} {eerror} {hC} {hCerror} {mag} {magError} {suss} {sussError}\n')

    energyAvgs.append(e)
    energyErrors.append(eerror)
    heatCaps.append(hC)
    hCerrors.append(hCerror)
    magAvgs.append(mag)
    magErrors.append(magError)
    susses.append(suss)
    sussErrors.append(sussError)
f.close()


fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes[0, 0].scatter(temps, magAvgs, marker='o', color='IndianRed')
axes[0, 0].set_xlabel('Value of kT')
axes[0, 0].set_ylabel('Total magnetization')
axes[0, 0].errorbar(temps, magAvgs, yerr=magErrors, fmt='o')

axes[0, 1].scatter(temps, susses, marker='o', color='RoyalBlue')
axes[0, 1].set_xlabel('Value of kT')
axes[0, 1].set_ylabel('Value of susceptibility')
axes[0, 1].errorbar(temps, susses, yerr=sussErrors, fmt='o')

axes[1, 0].scatter(temps, energyAvgs, marker='o', color='IndianRed')
axes[1, 0].set_xlabel('Value of kT')
axes[1, 0].set_ylabel('Total energy')
axes[1, 0].errorbar(temps, energyAvgs, yerr=energyErrors, fmt='o')

axes[1, 1].scatter(temps, heatCaps, marker='o', color='RoyalBlue')
axes[1, 1].set_xlabel('Value of kT')
axes[1, 1].set_ylabel('Heat capacity per spin')
axes[1, 1].errorbar(temps, heatCaps, yerr=hCerrors, fmt='o')

plt.tight_layout()
plt.savefig('Task4_witherror_fig')
plt.close()