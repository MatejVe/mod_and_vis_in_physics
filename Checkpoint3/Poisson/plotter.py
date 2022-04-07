import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_center_charge_potential():
    f = open('Checkpoint3/Poisson/center_charge.dat', 'r')
    data = f.readlines()
    f.close()
    N = int(round(np.power(len(data), 1/3), 0))
    potential = np.zeros(shape=(N,N,N))
    for line in tqdm(data):
        line = line.strip().split(' ')
        i, j, k = int(line[0]), int(line[1]), int(line[2])
        potential[i,j,k] = float(line[3])
    
    plt.figure(figsize=(10,8))
    plt.imshow(potential[:,:,int(N/2)], cmap='hot', interpolation='bilinear')
    plt.colorbar()
    plt.title('Potential of a 3D single charge \nin the middle of the box simulation')
    plt.tight_layout()
    plt.savefig('Checkpoint3/Poisson/center_charge_potential')
    plt.close()

def plot_potential_vs_distance_loglog():
    f = open('Checkpoint3/Poisson/center_charge_distance.dat', 'r')
    data = f.readlines()
    f.close()
    distances = []
    potentials = []
    for line in tqdm(data):
        line = line.strip().split(' ')
        distances.append(float(line[0]))
        potentials.append(float(line[1]))
    f.close()
    
    plt.figure(figsize=(10,8))
    plt.scatter(distances, potentials, marker='.')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Potential vs absolute distance from the center charge')
    plt.xlabel('Distance')
    plt.ylabel('Potential')
    plt.savefig('Checkpoint3/Poisson/center_charge_distance')
    plt.close()

def plot_efield_arrows_cut(z=50):
    f = open('Checkpoint3/Poisson/center_charge.dat', 'r')
    data = f.readlines()
    N = int(round(np.power(len(data), 1/3), 0))
    xs, ys, Exs, Eys = [], [], [], []
    for line in tqdm(data):
        line = line.strip().split(' ')
        k = int(line[2])
        if k == z:
            xs.append(int(line[0]))
            ys.append(int(line[1]))
            Exs.append(float(line[4]))
            Eys.append(float(line[5]))
    xs = np.array(xs).reshape((N,N))
    ys = np.array(ys).reshape((N,N))
    Exs = np.array(Exs).reshape((N,N))
    Eys = np.array(Eys).reshape((N,N))
    
    plt.figure(figsize=(8,8))
    plt.quiver(xs, ys, Exs, Eys, angles='xy', scale=1.2)
    plt.savefig('Checkpoint3/Poisson/efield_arrow_cutz50')

plot_efield_arrows_cut()