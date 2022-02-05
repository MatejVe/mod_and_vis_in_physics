from hashlib import new
import matplotlib
from pyparsing import col
matplotlib.use('TKAgg')

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def glauber_energy_change(spins, spinFlipLoc):
    # Assume the energy constant J = 1
    # Final delta E is 2J(S_t + S_l + S_r + S_b)*S_m
    rowloc, colloc = spinFlipLoc

    lrows, lcols = spins.shape

    # Find the nearest neighbours: [(upper), (left), (right), (lower)] (periodic boundary conditions apply)
    nearestNeigbours = [((rowloc - 1) % lrows, colloc), 
                        (rowloc, (colloc - 1) % lcols),
                        (rowloc, (colloc + 1) % lcols),
                        ((rowloc + 1) % lrows, colloc)]

    oldSpin = spins[rowloc, colloc]
    newSpin = -oldSpin
    neighbourSpinSum = 0
    for neigbour in nearestNeigbours:
        row, col = neigbour
        neighbourSpinSum += spins[row, col]

    return 2*neighbourSpinSum*oldSpin  # TODO: check if this is always true


def metropolis_test_true(deltaE, temp):
    if deltaE <= 0:
        return True
    else:
        prob = np.exp(-deltaE / temp)
        if prob >= np.random.random():
            return True
        else:
            return False

J=1.0
nstep=10000

#input

if(len(sys.argv) != 3):
    print ("Usage python ising.animation.py N T")
    sys.exit()

lx=int(sys.argv[1]) 
ly=lx 
kT=float(sys.argv[2]) 

spin=np.zeros((lx,ly),dtype=float)

#initialise spins randomly

for i in range(lx):
    for j in range(ly):
        r=random.random()
        if(r<0.5): spin[i,j]=-1
        if(r>=0.5): spin[i,j]=1

fig = plt.figure()
im=plt.imshow(spin, animated=True)

#update loop here - for Glauber dynamics

for n in range(nstep):
    for i in range(lx):
        for j in range(ly):

            #select spin randomly
            itrial=np.random.randint(0,lx)
            jtrial=np.random.randint(0,ly)
            spin_new=-spin[itrial,jtrial]

            #compute delta E eg via function (account for periodic BC)
            deltaE = glauber_energy_change(spin, (itrial, jtrial))

            #perform metropolis test
            if metropolis_test_true(deltaE, kT):
                spin[itrial, jtrial] = spin_new
                
    #occasionally plot or update measurements, eg every 10 sweeps
    if(n%10==0): 
#       update measurements
#       dump output
        f=open('spins.dat','w')
        for i in range(lx):
            for j in range(ly):
                f.write('%d %d %lf\n'%(i,j,spin[i,j]))
        f.close()
#       show animation
        plt.cla()
        im=plt.imshow(spin, animated=True)
        plt.draw()
        plt.pause(0.0001)