import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit 
import sys

if len(sys.argv) != 6:
    raise Exception('Parameters need to be: N KAPPA SIGMA v0 nstep')

N = int(sys.argv[1])
KAPPA = float(sys.argv[2])
SIGMA = float(sys.argv[3])
V0 = float(sys.argv[4])
nstep = int(sys.argv[5])

D = 1
DX = 1
DT = 0.01

vels = np.zeros(shape=(N, N))
for i in range(N):
    for j in range(N):
        vels[i,j] = -V0 * np.sin(2*np.pi*i/N)

@jit(nopython=True)
def xGrad(lat):
    xgrad = np.copy(lat)
    for i in range(N):
        for j in range(N):
            xgrad[i,j] = 1/DX * (lat[i,(j+1)%N] - lat[i,j])

    return xgrad

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
    xgrad = xGrad(lat)
    newlat = np.copy(lat)
    for i in range(N):
        for j in range(N):
            newlat[i,j] = lat[i,j] + D*DT/(DX**2) * (lat[(i+1)%N, j]
                                                    + lat[(i-1)%N, j]
                                                    + lat[i, (j+1)%N]
                                                    + lat[i, (j-1)%N]
                                                    - 4*lat[i,j]) \
                                                    + DT*RHO[i,j] \
                                                    - DT*KAPPA*lat[i,j] \
                                                    - DT*vels[i,j] * xgrad[i,j]
    return newlat

def simulate(lat, nstep):
    for n in tqdm(range(nstep)):
        lat = update_lattice(lat)
    plt.figure()
    plt.imshow(lat, interpolation='bilinear', cmap='bwr')
    plt.title(f'$v_0$={V0}')
    plt.savefig(f'Exam_practice/drift_{str(V0).replace(".", "")}')
    plt.close()

simulate(init_state(N), nstep)