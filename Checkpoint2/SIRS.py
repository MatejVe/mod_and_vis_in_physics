from subprocess import call
import matplotlib
matplotlib.use('TKAgg')

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors

def nearest_neighbours(row, col, latticeLen):
    return [((row - 1) % latticeLen, col), 
            (row, (col - 1) % latticeLen), 
            (row, (col + 1) % latticeLen), 
            ((row + 1) % latticeLen, col)]

#def nearest_neighbours(row, col, latticeLen):
#    return [((row - 1) % latticeLen, (col - 1) % latticeLen), # One up, one left
#            ((row - 1) % latticeLen, col),                    # One up
#            ((row - 1) % latticeLen, (col + 1) % latticeLen), # One up, one right
#            (row, (col - 1) % latticeLen),                    # One left
#            (row, (col + 1) % latticeLen),                    # One right
#            ((row + 1) % latticeLen, (col - 1) % latticeLen), # One down, one left
#            ((row + 1) % latticeLen, col),                    # One down
#            ((row + 1) % latticeLen, (col + 1) % latticeLen)] # One down, one right

class SIRS():

    def __init__(self, N, p1, p2, p3, initState='random', immune=0):
        
        self.N = N
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        if initState == 'random':
            self.lattice = []
            for i in range(N):
                row = []
                for j in range(N):
                    row.append(np.random.choice([1, 2, 0]))
                self.lattice.append(row)
        elif initState == 'outbreak':
            self.lattice = np.ones((N, N))
            self.lattice[[0,0,1,1,2,2,2,0,1], [0,1,0,1,0,1,2,2,2]] = 2
        else:
            self.lattice = initState
        self.lattice = np.array(self.lattice)

        immuneNum = round(immune*N**2)
        for i in range(immuneNum):
            irand = np.random.randint(0, N)
            jrand = np.random.randint(0, N)
            self.lattice[irand, jrand] = 3

    def sweep(self):
        for i in range(self.N):
                for j in range(self.N):
                    itrial=np.random.randint(0, self.N)
                    jtrial=np.random.randint(0, self.N)

                    if self.lattice[itrial, jtrial] == 1: # 'S'
                        Inum = 0
                        nearNeigh = nearest_neighbours(itrial, jtrial, self.N)
                        for neighbour in nearNeigh:
                            if self.lattice[neighbour] == 2: # 'I'
                                Inum += 1
                        if Inum > 0 and self.p1 > np.random.random():
                            self.lattice[itrial, jtrial] = 2 # 'I'
                    elif self.lattice[itrial, jtrial] == 2: # 'I'
                        if self.p2 > np.random.random():
                            self.lattice[itrial, jtrial] = 0 # 'R'
                    elif self.lattice[itrial, jtrial] == 0: # 'R'
                        if self.p3 > np.random.random():
                            self.lattice[itrial, jtrial] = 1 # 'S'        

    def visualise(self, nstep=100):
        # Encode the states as following: Susceptible - 1, Infected - 2, Recovering - 0
        colorsToUse = {'S':'y', 'I':'r', 'R':'g'}
        cmap = colors.ListedColormap(['g', 'y', 'r'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure(figsize=(8,8))
        im = plt.imshow(self.lattice, animated=True, cmap=cmap, norm=norm)
        plt.colorbar(im, cmap=cmap, norm=norm, boundaries=bounds, ticks=[0, 1, 2])

        for n in range(nstep):
            self.sweep()
            plt.cla()
            im=plt.imshow(self.lattice, animated=True, cmap=cmap, norm=norm)
            plt.draw()
            plt.pause(0.1)

    def infectiveness_measure(self, sweeps, callibration):
        infectedRatios = []
        for n in range(sweeps):
            self.sweep()
            if n > callibration:
                inf = np.count_nonzero(self.lattice == 2) / self.N**2
                if inf == 0:
                    return 0
                infectedRatios.append(inf)
        avg = np.mean(infectedRatios)
        return avg

    def infectiveness_variance_measure(self, sweeps, callibration):
        iR = []
        iR2 = []
        for n in range(sweeps):
            self.sweep()
            if n > callibration:
                inf = np.count_nonzero(self.lattice == 2)
                if inf == 0:
                    return 0
                iR.append(inf)
                iR2.append(inf**2)
        return (np.mean(iR2) - np.mean(iR)**2) / self.N**2

    def infectiveness_variance_measure_with_error(self, sweeps, callibration, measureInterval):
        iR = []
        iR2 = []
        for n in range(sweeps):
            self.sweep()
            if n > callibration and n % measureInterval == 0:
                inf = np.count_nonzero(self.lattice == 2)
                if inf == 0:
                    return (0, 0)
                iR.append(inf)
                iR2.append(inf**2)
        meaniR = np.mean(iR)
        meaniR2 = np.mean(iR2)
        var = (meaniR2 - meaniR**2) / self.N**2

        # Jacknife errors
        error = []
        for i in range(len(iR)):
            newiR = iR[:i] + iR[i+1:]
            newiR2 = iR2[:i] + iR2[i+1:]
            newmeaniR = np.mean(newiR)
            newmeaniR2 = np.mean(newiR2)
            newvar = (newmeaniR2 - newmeaniR**2) / self.N**2
            error.append((var - newvar)**2)
        error = np.sqrt(np.sum(error))
        return (var, error)


def only_susceptible_remain():
    s = SIRS(50)
    s.visualise(0.1, 0.8, 0.8)

def dynamic_equilibrium_SIR():
    s = SIRS(50)
    s.visualise(0.5, 0.5, 0.5)

def cyclic_wave():
    s = SIRS(100, 0.8, 0.1, 0.01, initState='outbreak')
    s.visualise(nstep=500)

s = SIRS(100, 0.8, 0.1, 0.01, initState='outbreak')
s.visualise(nstep=300)