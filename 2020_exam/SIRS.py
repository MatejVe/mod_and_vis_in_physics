from turtle import update
import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
import sys

if len(sys.argv) != 4:
    raise Exception('Parameters need to be: L p nstep')

L = int(sys.argv[1])
P = float(sys.argv[2])
nstep = int(sys.argv[3])

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

def animate(lat, nstep):
    plt.figure()
    plt.imshow(lat, animated=True)
    plt.colorbar()

    for n in tqdm(range(nstep)):
        lat = sweep(lat, P)

        plt.cla()
        plt.title(n)
        plt.imshow(lat, interpolation='bilinear', animated=True)
        plt.draw()
        plt.pause(0.001)

def fraction_active(lat):
    return np.sum(lat)/L**2

def infected_vs_time_plot():
    lattice = initial_state()
    actives = []
    iters = []
    for n in tqdm(range(nstep)):
        lattice = sweep(lattice, P)
        actives.append(fraction_active(lattice))
        iters.append(n)
    
    plt.figure(figsize=(10,8))
    plt.plot(iters, actives)
    plt.xlabel('Sweeps')
    plt.ylabel('Fraction infected')
    plt.title(f'$p=${P}')
    plt.savefig(f'2020_exam/fraction_vs_sweeps_{str(P).replace(".","")}')
    plt.close()

def fraction_of_active_sites_plot():
    fracs = []
    ps = np.linspace(0.55, 0.7, 30)

    for p in tqdm(ps):
        lat = initial_state()
        actives = []
        for n in range(nstep):
            lat = sweep(lat, p)
            actives.append(np.sum(lat))
        fracs.append(np.mean(actives)/L**2)

    plt.figure(figsize=(10,8))
    plt.plot(ps, fracs)
    plt.xlabel('p value')
    plt.ylabel('Average infected sites')
    plt.savefig('2020_exam/infected_vs_p')
    plt.close()

def variance_in_active_cells():
    ps = np.linspace(0.55, 0.7, 30)

    vars = []
    errors = []
    for p in tqdm(ps):
        lat = initial_state()
        actives = []
        actives2 = []
        for n in range(nstep):
            lat = sweep(lat, p)
            actives.append(np.sum(lat))
            actives2.append(np.sum(lat)**2)
        
        var = (np.mean(actives2) - np.mean(actives)**2)/L**2
        vars.append(var)
        temperrs = []
        for n in range(nstep):
            nActives = actives[:n] + actives[n+1:]
            nActives2 = actives2[:n] + actives2[n+1:]
            newvar = (np.mean(nActives2) - np.mean(nActives)**2)/L**2
            temperrs.append((newvar - var)**2)
        errors.append(np.sqrt(np.sum(temperrs)))


    plt.figure(figsize=(10,8))
    plt.errorbar(ps, vars, errors)
    plt.xlabel('Value of p')
    plt.ylabel('Fluctuation measure value')
    plt.savefig('2020_exam/fluctuation_vs_p')
    plt.close()

def survival_probability(p, nstep):
    niter = 30
    survivors = []
    for i in range(niter):
        lat = np.zeros(shape=(L,L))
        lat[int(L/2),int(L/2)] = 1
        for j in range(nstep):
            lat = sweep(lat, p)
        survivors.append(np.sum(lat))
    return np.mean(survivors)

def plot_survival_vs_sweeps(p):
    ts = np.linspace(50, 300, 26)
    survivalbilities = []
    for t in tqdm(ts):
        survivalbilities.append(survival_probability(p, int(t)))
    
    plt.figure(figsize=(10,8))
    plt.plot(ts, survivalbilities)
    plt.xlabel('Sweeps')
    plt.ylabel('Cells remaining')
    plt.title(f'$p=${p}')
    plt.savefig(f'2020_exam/survivors_vs_time_{str(p).replace(".","")}')
    plt.close()

variance_in_active_cells()