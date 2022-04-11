import numpy as np
from tqdm import tqdm
import sys
from scipy.stats import multivariate_normal

if len(sys.argv) != 4:
    raise Exception('Parameters need to be: N atol charge-distribution (options: center, random)')

N = int(sys.argv[1])
atol = float(sys.argv[2])

potential = np.zeros((N,N,N))
cdist = sys.argv[3]
if cdist == 'center':
    charge = np.zeros((N,N,N))
    charge[int(N/2),int(N/2),int(N/2)] = 1
elif cdist == 'random':
    charge = np.random.uniform(0,1,size=(N-30,N-30,N-30))
    charge = np.pad(charge, 15)
#elif cdist == 'gaussian':
#    x = np.arange(N**3).reshape((N,N,N))


def jacobi_update(potential, chargedist):
    newPot = 1/6 * (np.roll(potential, 1, axis=0)
                    +np.roll(potential, -1, axis=0)
                    +np.roll(potential, 1, axis=1)
                    +np.roll(potential, -1, axis=1)
                    +np.roll(potential, 1, axis=2)
                    +np.roll(potential, -1, axis=2)
                    +chargedist)
    
    # Ensure the boundary is 0
    return np.pad(newPot[1:N-1,1:N-1,1:N-1], ((1,1),(1,1),(1,1)), 'constant', constant_values=(0))

# update potential once
oldPotential = np.copy(potential)
potential = jacobi_update(potential, charge)

iteration = 1
while(np.any(np.abs(potential - oldPotential) > atol)):
    oldPotential = np.copy(potential)
    potential = jacobi_update(potential, charge)
    iteration += 1
    print(f'Iteration #{iteration}.')

Ex, Ey, Ez = np.gradient(potential)
Ex *= -1
Ey *= -1
Ez *= -1

f = open('Checkpoint3/Poisson/center_charge.dat', 'w')
print('Writing down the potential at the lattice points to the center_charge.dat')

for i in range(N):
    for j in range(N):
        for k in range(N):
            f.write(f'{i} {j} {k} {potential[i,j,k]} {Ex[i,j,k]} {Ey[i,j,k]} {Ez[i,j,k]}\n')
f.close()

f = open('Checkpoint3/Poisson/center_charge_distance.dat', 'w')
print('Writing down the distances, potential and electric field to the center_charge_distance.dat')

distance = []
pot = []
for i in tqdm(range(N)):
    for j in range(N):
        for k in range(N):
            d = np.sqrt((i-N/2)**2+(j-N/2)**2+(k-N/2)**2)
            distance.append(d)
            pot.append(potential[i,j,k])
            f.write(f'{d} {potential[i,j,k]} {np.sqrt(Ex[i,j,k]**2+Ey[i,j,k]**2+Ez[i,j,k]**2)}\n')
f.close()