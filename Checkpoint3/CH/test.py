from os import system
import sys
import matplotlib
matplotlib.use('TKAgg')
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as pb

nstep = int(1e+6)+1
N = 100
M = 0.1
a = 0.1
k = 0.1
phi0 = 0.
dt = 2.
dx = 1.

#calc
r_1 = k/dx**2
r_2 = M*(dt/dx**2)
r_3 = 0.5*r_1


system = np.random.uniform(-0.1+phi0,0.1+phi0,(N,N))

    
def update():
    mu = -a*system + a*np.power(system,3) - (r_1)*(np.roll(system,1,axis=0) + 
                                                             np.roll(system,-1,axis=0) + 
                                                             np.roll(system,1,axis=1) + 
                                                             np.roll(system,-1,axis=1)-np.multiply(system,4))
    return r_2*( np.roll(mu,1,axis=0) + 
                        np.roll(mu,-1,axis=0) + 
                        np.roll(mu,1,axis=1) + 
                        np.roll(mu,-1,axis=1)-np.multiply(mu,4) ) 
   
def fe():
    return np.sum(-(a/2) * np.power(system,2)+(a/4)*np.power(system,4)+r_3*(np.power(np.roll(system,1,axis=0) - 
                                                             system,2) + 
                                                             np.power(np.roll(system,1,axis=1) - 
                                                             system,2))    )

plt.figure()
plt.imshow(system,vmax=1,vmin=-1, animated=True,cmap='ocean_r') 
plt.colorbar()
f = np.empty((nstep//100)+1)

for i in pb(range(nstep)):
    system += update()
    
    if i%100 == 0:

        f[i//100] = fe()
        plt.cla()   
        plt.title(i)
        plt.imshow(system,vmax=1,vmin=-1,animated=True,interpolation='bilinear',cmap='ocean_r')
        plt.draw()
        plt.pause(0.0001)
        
np.savetxt("Checkpoint_3/CH/data/output_new.dat",f)