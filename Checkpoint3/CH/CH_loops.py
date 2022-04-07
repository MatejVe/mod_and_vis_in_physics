import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from numba import jit

A = 0.1
M = 0.1
KAPPA = 0.1
DX = 1
DT = 2

@jit(nopython=True)
def mu(lat, i, j):
    N = len(lat)
    return -A*lat[i,j] + A*lat[i,j]**3 - KAPPA/(DX)**2 * (lat[(i+1)%N, j] + lat[i-1, j] + lat[i, (j+1)%N] + lat[i, j-1] - 4 * lat[i, j])

@jit(nopython=True)
def update_lattice_loops(lat):
    N = len(lat)
    newlat = np.copy(lat)
    for i in range(N):
        for j in range(N):
            m1 = mu(lat, (i+1)%N, j)
            m2 = mu(lat, i-1, j)
            m3 = mu(lat, i, (j+1)%N)
            m4 = mu(lat, i, j-1)
            m5 = mu(lat, i, j)
            newlat[i, j] = lat[i, j] + M*DT/(DX)**2 * (m1+m2+m3+m4-4*m5)
    return newlat

def animate(lattice, nstep):
    plt.figure()
    plt.imshow(lattice, vmax=1, vmin=-1, animated=True, cmap='bwr')
    plt.colorbar()

    for n in tqdm(range(nstep)):
        lattice = update_lattice_loops(lattice)

        if n%50 == 0:
            plt.cla()
            plt.title(n)
            plt.imshow(lattice, vmax=1, vmin=.1, interpolation='bilinear', animated=True, cmap='bwr')
            plt.draw()
            plt.pause(0.00001)

def init_const_noise(size, const):
    lat = np.random.uniform(const-0.1, const+0.1, size=(size, size))
    return lat

lattice = init_const_noise(100, 0.5)

animate(lattice, nstep=100000)