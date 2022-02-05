import matplotlib
matplotlib.use('TKAgg')

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Important functions used in the code

def are_neighbours(row1, col1, row2, col2, latticeLen):
    # TODO: check if this works, see the papers
    if abs(row2 - row1) + abs(col2 - col1) == 1:
        return True
    if abs(row2 - row1) + abs(col2 - col1) == latticeLen - 1:
        if abs(row2 - row1) == 0 or abs(col2 - col1) == 0:
            return True
    return False

def nearest_neighbours(row, col, latticeLen):
    return [((row - 1) % latticeLen, col), 
            (row, (col - 1) % latticeLen), 
            (row, (col + 1) % latticeLen), 
            ((row + 1) % latticeLen, col)]

def kawasaki_energy_change(spins, spinFlipLocs):
    # Assume the energy constant J = 1 and that the two spins in question have different orientations (+1 and -1)
    row1, col1, row2, col2 = spinFlipLocs
    spin1, spin2 = spins[row1, col1], spins[row2, col2]

    lrows, lcols = spins.shape

    deltaE = 0
    if are_neighbours(row1, col1, row2, col2, lrows):
        for neighbour in nearest_neighbours(row1, col1, lrows):
            row, col = neighbour
            if row != row2 and col != col2:
                deltaE += spins[row2, col2] * spins[row, col]
        for neighbour in nearest_neighbours(row2, col2, lrows):
            row, col = neighbour
            if row != row1 and col != col2:
                deltaE += spins[row1, col1] * spins[row, col]
    else:
        for neighbour in nearest_neighbours(row1, col1, lrows):
            row, col = neighbour
            deltaE += spins[row2, col2] * spins[row, col]
        for neighbour in nearest_neighbours(row2, col2, lrows):
            row, col = neighbour
            deltaE += spins[row1, col1] * spins[row, col]
    return -2*deltaE

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

if(len(sys.argv) != 4):
    print("Usage python ising.animation.py N T mode")
    sys.exit()

lx=int(sys.argv[1])
ly=lx
kT=float(sys.argv[2])
mode=str(sys.argv[3])

spin=np.zeros((lx,ly),dtype=float)

#initialise spins randomly

for i in range(lx):
    for j in range(ly):
        r=random.random()
        if(r<0.5): spin[i,j]=-1
        if(r>=0.5): spin[i,j]=1
    
fig = plt.figure()
im=plt.imshow(spin, animated=True)

# Start simulating

for n in range(nstep):
    for i in range(lx):
        for j in range(ly):
            # start with Glauber dynamics
            if mode == 'glauber':
                #select spin randomly
                itrial=np.random.randint(0,lx)
                jtrial=np.random.randint(0,ly)
                spin_new=-spin[itrial,jtrial]
                
                #compute delta E eg via function (account for periodic BC)
                deltaE = glauber_energy_change(spin, (itrial, jtrial))

                #perform metropolis test
                if metropolis_test_true(deltaE, kT):
                    spin[itrial, jtrial] = spin_new
            elif mode == 'kawasaki':
                #select spins randomly
                i1=np.random.randint(0,lx)
                j1=np.random.randint(0,ly)
                spin1 = spin[i1, j1]
                i2=np.random.randint(0,lx)
                j2=np.random.randint(0,ly)
                spin2 = spin[i2, j2]

                if i1 != i2 and j1 != j2: # if the spins are at different positions
                    if spin1 != spin2:  # if the spins are different
                        #compute delta E eg via function (account for periodic BC)
                        deltaE = kawasaki_energy_change(spin, (i1, j1, i2, j2))

                        #perform metropolis test
                        if metropolis_test_true(deltaE, kT):
                            spin[i1, j1] = spin2
                            spin[i2, j2] = spin1
            else:
                raise Exception("Invalid mode")
                
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