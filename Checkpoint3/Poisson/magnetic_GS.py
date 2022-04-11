import numpy as np
from tqdm import tqdm
import sys
from numba import jit

if len(sys.argv) != 3:
    raise Exception('Parameters need to be: N atol')

N = int(sys.argv[1])
atol = float(sys.argv[2])

potential = np.zeros((N,N,N))
currents = np.zeros((N,N,N))
currents[int(N/2), int(N/2),:] = 1

@jit(nopython=True)
def gauss_seidel_update(potential, chargedist):
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                potential[i,j,k] = 1/6 * (potential[(i+1)%N,j,k]
                                            + potential[i-1,j,k]
                                            + potential[i,(j+1)%N,k]
                                            + potential[i,j-1,k]
                                            + potential[i,j,(k+1)%N]
                                            + potential[i,j,k-1]
                                            + chargedist[i,j,k])
    return potential

oldPotential = np.copy(potential)
potential = gauss_seidel_update(potential, currents)

iteration = 1
while(np.any(np.abs(potential - oldPotential) > atol)):
    oldPotential = np.copy(potential)
    potential = gauss_seidel_update(potential, currents)
    iteration += 1
    print(f'Iteration #{iteration}.')

Bx = np.roll(potential, 1, axis=1) - potential
By = -np.roll(potential, 1, axis=0) + potential

f = open('Checkpoint3/Poisson/one_current.dat', 'w')
print('Writing down the indices, the vector potential and the magnetic field components to the one_current.dat')
for i in tqdm(range(N)):
    for j in range(N):
        for k in range(N):
            f.write(f'{i} {j} {k} {potential[i,j,k]} {Bx[i,j,k]} {By[i,j,k]}\n')
f.close()