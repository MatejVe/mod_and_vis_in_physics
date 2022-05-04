import matplotlib
matplotlib.use('TKAgg')

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from auxiliary_functions import *

def hfunc(i,j,n):
    P = 10
    TAU = 10000
    H0 = 10
    return H0*np.cos(2*np.pi*i/P)*np.cos(2*np.pi*j/P)*np.sin(2*np.pi*n/TAU)

def glauber_energy_change(spins, spinFlipLoc, j, h, n):
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

    return 2*j*neighbourSpinSum*oldSpin + 2*h(rowloc,colloc,n)*oldSpin

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

def sweep(lat, j, h, kt, l, n):
    for i in range(l):
        for k in range(l):
            itrial=np.random.randint(0,l)
            jtrial=np.random.randint(0,l)
            deltaE = glauber_energy_change(lat, (itrial, jtrial), j, h, n)
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

def energy(lat, j, h, n):
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
            energy += -h(i,j,n)*lat[i,j]
    return energy

L = 50
J = -1
KT = 1

lat = init_state(L)
times = []
maxfield = []
magstag = []
for n in tqdm(range(7500)):
    lat = sweep(lat, J, hfunc, KT, L, n)

    if n%100==0:
        times.append(n)
        maxfield.append(10*np.sin(2*np.pi*n/10000))
        magstag.append(stagmag(lat))

scatter_and_save_data(times, maxfield, '2016_exam/maxfield_P10', 'Time', 'Max field')
scatter_and_save_data(times, magstag, '2016_exam/magstag_P10', 'Time', 'Staggered magnetisation')
