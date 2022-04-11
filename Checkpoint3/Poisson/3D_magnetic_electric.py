import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def magnetic_3D():
    f = open('Checkpoint3/Poisson/one_current.dat', 'r')
    data = f.readlines()
    N = int(round(np.power(len(data), 1/3), 0))
    xs, ys, zs, Bxs, Bys, Bzs = [], [], [], [], [], []
    for line in tqdm(data):
        line = line.strip().split(' ')
        i = int(line[0])
        xs.append(i)
        j = int(line[1])
        ys.append(j)
        k = int(line[2])
        zs.append(k)
        Bx = float(line[4])
        Bxs.append(Bx)
        By = float(line[5])
        Bys.append(By)
        Bzs.append(0)
        
    N = int(N)
    xs = np.array(xs).reshape((N,N,N))[48:53,48:53,48:53]
    ys = np.array(ys).reshape((N,N,N))[48:53,48:53,48:53]
    zs = np.array(zs).reshape((N,N,N))[48:53,48:53,48:53]
    Bxs = np.array(Bxs).reshape((N,N,N))[48:53,48:53,48:53]
    Bys = np.array(Bys).reshape((N,N,N))[48:53,48:53,48:53]
    Bzs = np.array(Bzs).reshape((N,N,N))[48:53,48:53,48:53]

    Bxs = Bxs / np.sqrt(np.sum(np.power(Bxs, 2)))
    Bys = Bys / np.sqrt(np.sum(np.power(Bys, 2)))
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(xs,ys,zs,Bxs,Bys,Bzs)#, length=0.1, normalize=True)
    plt.savefig('Checkpoint3/Poisson/3D_magnetic_field')
    plt.close()

def electric_3D():
    f = open('Checkpoint3/Poisson/center_charge.dat', 'r')
    data = f.readlines()
    N = int(round(np.power(len(data), 1/3), 0))
    xs, ys, zs, Exs, Eys, Ezs = [], [], [], [], [], []
    for line in tqdm(data):
        line = line.strip().split(' ')
        xs.append(int(line[0]))
        ys.append(int(line[1]))
        zs.append(int(line[2]))
        Exs.append(float(line[4]))
        Eys.append(float(line[5]))
        Ezs.append(float(line[6]))
    xs = np.array(xs).reshape((N,N,N))[48:53,48:53,48:53]
    ys = np.array(ys).reshape((N,N,N))[48:53,48:53,48:53]
    zs = np.array(zs).reshape((N,N,N))[48:53,48:53,48:53]
    Exs = np.array(Exs).reshape((N,N,N))[48:53,48:53,48:53]
    Eys = np.array(Eys).reshape((N,N,N))[48:53,48:53,48:53]
    Ezs = np.array(Ezs).reshape((N,N,N))[48:53,48:53,48:53]

    Exs = Exs / np.sqrt(np.sum(np.power(Exs, 2)))
    Eys = Eys / np.sqrt(np.sum(np.power(Eys, 2)))
    Ezs = Ezs / np.sqrt(np.sum(np.power(Ezs, 2)))
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(xs,ys,zs,Exs,Eys,Ezs)
    plt.savefig('Checkpoint3/Poisson/3D_electric_field')
    plt.close()

electric_3D()
magnetic_3D()