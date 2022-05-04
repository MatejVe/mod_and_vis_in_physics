import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit 
import sys

if len(sys.argv) != 5:
    raise Exception('Parameters need to be: N KAPPA SIGMA nstep')

N = int(sys.argv[1])
KAPPA = float(sys.argv[2])
SIGMA = float(sys.argv[3])
nstep = int(sys.argv[4])

D = 1
DX = 1
DT = 0.1

RHO = np.zeros(shape=(N,N))
for i in range(N):
    for j in range(N):
        RHO[i, j] = np.exp(-((i-int(N/2))**2+(j-int(N/2))**2)/SIGMA**2)

def init_state(N):
    phi = np.ones(shape=(N,N)) * 0.5 + np.random.uniform(-0.1, 0.1, size=(N,N))
    return phi

@jit(nopython=True)
def update_lattice(lat):
    N = len(lat)
    newlat = np.copy(lat)
    for i in range(N):
        for j in range(N):
            newlat[i,j] = lat[i,j] + D*DT/(DX**2) * (lat[(i+1)%N, j]
                                                    + lat[(i-1)%N, j]
                                                    + lat[i, (j+1)%N]
                                                    + lat[i, (j-1)%N]
                                                    - 4*lat[i,j]) \
                                                    + DT*RHO[i,j] \
                                                    - DT*KAPPA*lat[i,j]
    return newlat

def av_phi(lat):
    return np.average(lat)

def animate(lat, nstep):
    plt.figure()
    plt.imshow(lat, animated=True, cmap='bwr')
    plt.colorbar()

    f = open('Exam_practice/av_phi.dat', 'w')

    for n in tqdm(range(nstep)):
        lat = update_lattice(lat)

        if n%100 == 0:
            avPhi = av_phi(lat)
            f.write(f'{avPhi} {n}\n')

            plt.cla()
            plt.title(n)
            plt.imshow(lat, interpolation='bilinear', animated=True, cmap='bwr')
            plt.draw()
            plt.pause(0.00001)
    f.close()

    f = open('Exam_practice/phi_vs_distance.dat', 'w')
    for i in range(N):
        for j in range(N):
            f.write(f'{i} {j} {lat[i,j]}\n')
    f.close()

lattice = init_state(N)

animate(lattice, nstep)