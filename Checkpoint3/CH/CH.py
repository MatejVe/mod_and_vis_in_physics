import matplotlib

matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

M = 0.1
A = 0.1
KAPPA = 0.1

def update_lattice(lat, dt, dx):
    N = len(lat)
    mu = -A*lat + A*np.power(lat,3) - KAPPA/(dx)**2 * (np.roll(lat, 1, axis=1) 
                                                    + np.roll(lat, -1, axis=1)
                                                    + np.roll(lat, 1, axis=0) 
                                                    + np.roll(lat, -1, axis=0) 
                                                    - np.multiply(4, lat))
    
    newlat = lat + M*dt/(dx)**2 * (np.roll(mu, 1, axis=1)
                                + np.roll(mu, -1, axis=1)
                                + np.roll(mu, 1, axis=0)
                                + np.roll(mu, -1, axis=0)
                                - np.multiply(4, mu))
    return newlat

def animate(lattice, dx, dt, nstep):
    plt.figure()
    plt.imshow(lattice, vmax=1, vmin=-1, animated=True, cmap='bwr')
    plt.colorbar()

    for n in tqdm(range(nstep)):
        lattice = update_lattice(lattice, dt, dx)

        if n%50 == 0:
            plt.cla()
            plt.title(n)
            plt.imshow(lattice, vmax=1, vmin=.1, interpolation='bilinear', animated=True, cmap='bwr')
            plt.draw()
            plt.pause(0.00001)


def init_const_noise(size, const):
    lat = np.random.uniform(const-0.1, const+0.1, size=(size, size))
    return lat

def free_energy(lat, dx):
    return np.sum(-A/2 * np.power(lat, 2) + A/4 * np.power(lat, 4) + KAPPA/(2*dx**2) * (np.power(np.roll(lat, 1, axis=0)-lat, 2) + np.power(np.roll(lat, 1, axis=1)-lat, 2)))

def plot_fe(lat, dx, dt, nstep, filename):
    f = open('Checkpoint3/' + filename +'.dat', 'w')
    fes = []
    ajs = []
    for i in tqdm(range(nstep)):
        lat = update_lattice(lat, dt, dx)
        if i%100 == 0:
            fe = free_energy(lat, dx=dx)
            ajs.append(i)
            fes.append(fe)
            f.write(f'{i}-{fe}\n')
    f.close()
    plt.figure()
    plt.plot(ajs, fes)
    plt.title('Free energy density over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Free Energy Density')
    plt.savefig('Checkpoint3/' + filename)
    plt.close()

lattice = init_const_noise(100, 0.5)
#plot_fe(lattice, dx=1, dt=2, nstep=1000000, filename='buble_fe')

animate(lattice, dx=1, dt=2, nstep=100000)