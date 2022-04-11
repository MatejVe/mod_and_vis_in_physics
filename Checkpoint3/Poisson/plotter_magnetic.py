import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def plot_one_wire_potential():
    f = open('Checkpoint3/Poisson/one_current.dat', 'r')
    data = f.readlines()
    N = int(round(np.power(len(data), 1/3), 0))
    potential = np.zeros(shape=(N,N,N))
    for line in tqdm(data):
        line = line.strip().split(' ')
        i, j, k = int(line[0]), int(line[1]), int(line[2])
        potential[i,j,k] = float(line[3])

    plt.figure(figsize=(10,8))
    plt.imshow(potential[:,:,int(N/2)], cmap='hot', interpolation='bilinear')
    plt.colorbar()
    plt.title('Potential of a 3D single wire current \nin the middle of the box simulation')
    plt.tight_layout()
    plt.savefig('Checkpoint3/Poisson/one_wire_potential')
    plt.close()

def plot_bfield_arrow_cut(mini=0, maxi=None, z=50):
    f = open('Checkpoint3/Poisson/one_current.dat', 'r')
    data = f.readlines()
    if maxi is None:
        N = int(round(np.power(len(data), 1/3), 0))
        maxi = N
    else:
        N = maxi-mini
    xs, ys, Bxs, Bys = [], [], [], []
    for line in tqdm(data):
        line = line.strip().split(' ')
        i = int(line[0])
        j = int(line[1])
        k = int(line[2])
        if k == z and mini<=i<maxi and mini<=j<maxi:
            xs.append(int(line[0]))
            ys.append(int(line[1]))
            Bxs.append(float(line[4]))
            Bys.append(float(line[5]))
    xs = np.array(xs).reshape((N,N))
    ys = np.array(ys).reshape((N,N))
    Bxs = np.array(Bxs).reshape((N,N))
    Bys = np.array(Bys).reshape((N,N))

    # normalize
    Bxs = Bxs / np.sqrt(np.sum(np.power(Bxs, 2)))
    Bys = Bys / np.sqrt(np.sum(np.power(Bys, 2)))
    plt.figure(figsize=(8,8))
    plt.quiver(xs, ys, Bxs, Bys, angles='xy', scale=1.2)
    plt.title('Magnetic field')
    plt.savefig('Checkpoint3/Poisson/bfield_arrows')
    plt.close()

plot_one_wire_potential()
plot_bfield_arrow_cut(40, 60)