import numpy as np
from tqdm import tqdm
from numba import jit
from scipy import optimize
import matplotlib.pyplot as plt

N = 100

@jit(nopython=True)
def gauss_seidel_or_update(potential, chargedist, w):
    for i in range(1, N-1):
        for j in range(1, N-1):
            for k in range(1, N-1):
                potential[i,j,k] = (1-w)*potential[i,j,k] + \
                                    w/6 * (potential[(i+1)%N,j,k]
                                            + potential[i-1,j,k]
                                            + potential[i,(j+1)%N,k]
                                            + potential[i,j-1,k]
                                            + potential[i,j,(k+1)%N]
                                            + potential[i,j,k-1]
                                            + chargedist[i,j,k])
    return potential

@jit(nopython=True)
def GS_or_iterations(w, atol=1e-5):
    potential = np.zeros((N,N,N))
    charge = np.zeros((N,N,N))
    charge[int(N/2),int(N/2),int(N/2)] = 1

    oldPotential = np.copy(potential)
    potential = gauss_seidel_or_update(potential, charge, w)

    iteration = 1
    while(np.any(np.abs(potential - oldPotential) > atol)):
        oldPotential = np.copy(potential)
        potential = gauss_seidel_or_update(potential, charge, w)
        iteration += 1
        if iteration > 200:
            return iteration

    return iteration

ws = np.linspace(1.5, 1.9, 100)
iterations = []

for w in tqdm(ws):
    iterations.append(GS_or_iterations(w))
print(f'Fastest convergence occurs for w={round(ws[np.argmin(iterations)],2)}')
plt.figure(figsize=(10,8))
plt.plot(ws, iterations)
plt.title('Number of iterations untill convergence vs over-relaxation parameter $w$')
plt.xlabel('Parameter $w$')
plt.ylabel('Iterations until convergence')
plt.savefig('Checkpoint3/Poisson/GS convergence')
plt.close()