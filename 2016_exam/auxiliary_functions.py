import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def scatter_and_save_data(x, y, filepathname, xlabel, ylabel, title=None, yerr=None):
    plt.figure(figsize=(10,8))
    if yerr is not None:
        plt.errorbar(x, y, yerr)
    else:
        plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filepathname)
    plt.close()

    f = open(filepathname + '.dat', 'w')
    if yerr is not None:
        for i1, i2, i3 in zip(x, y, yerr):
            f.write(f'{i1} {i2} {i3}\n')
    else:
        for i1, i2 in zip(x, y):
            f.write(f'{i1} {i2}\n')
    f.close()

def save_lat_img(lat, filepathname, title, interpolation=None):
    plt.figure(figsize=(10,10))
    plt.imshow(lat, interpolation=interpolation)
    plt.title(title)
    plt.savefig(filepathname)
    plt.close()

def animate(lat, nstep, updaterule, *args, upint=10):
    plt.figure()
    plt.imshow(lat, animated=True)
    plt.colorbar()

    for n in tqdm(range(nstep)):
        lat = updaterule(lat, *args)

        if n%upint==0:
            plt.cla()
            plt.title(n)
            plt.imshow(lat, animated=True)
            plt.draw()
            plt.pause(0.001)

def jacknife_error(arr):
    errors = []
    for i in range(len(arr)):
        newArr = arr[:i] + arr[i+1:]
        newMean = np.mean(newArr)
        errors.append((newMean - np.mean(arr))**2)
    return np.sqrt(np.sum(errors))

def response_function_jacknife_err(array, scale=1):
    array2 = [el**2 for el in array]
    var = (np.mean(array2) - np.mean(array)**2)/scale
    errs = []
    for i in range(len(array)):
        narray = array[:i] + array[i+1:]
        narray2 = array2[:i] + array2[i+1:]
        newvar = (np.mean(narray2) - np.mean(narray)**2)/scale
        errs.append((newvar - var)**2)
    err =np.sqrt(np.sum(errs))
    return var, err

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

def simulate_and_measure(lat, nstep, *args, 
                        equilibration=0, upint=10, std='std', **kwargs):
    """_summary_

    Args:
        lat (_type_): _description_
        nstep (_type_): _description_
        equilibration (int, optional): _description_. Defaults to 0.
        upint (int, optional): _description_. Defaults to 10.
        std (str, optional): _description_. Defaults to 'std'.
    """
    updaterule, measurefuncs = args[0], args[1:]

    measured = [[] for i in range(len(measurefuncs))]
    for n in tqdm(range(nstep)):
        lat = updaterule(lat, *kwargs[updaterule.__name__])
        if n>equilibration and n%upint==0:
            for i, func in enumerate(measurefuncs):
                measured[i].append(func(lat, *kwargs[func.__name__]))

    toreturn = []
    for measure in measured:   
        if std=='std':
            var = np.std(measure)
        elif std=='jacknife':
            var = jacknife_error(measure)
        toreturn.append((np.mean(measure), var))
    return toreturn
