import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
from numba import jit
from auxiliary_functions import *

# Set parameters
DX = 1
L = 50

def initial_state(l):
    lat = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            lat[i,j] = np.random.uniform(0, 1/3)
    return lat

@jit(nopython=True)
def update_concentrations(concs, d, p, q, dt):
    a, b, c = concs
    anew = np.copy(a)
    bnew = np.copy(b)
    cnew = np.copy(c)
    newconcs = [anew, bnew, cnew]
    for i in range(L):
        for j in range(L):
            for k in range(3):
                newconcs[k][i,j] = concs[k][i,j] + dt*(
                    d/DX**2 * (
                        concs[k][(i+1)%L,j] + concs[k][(i-1)%L,j] + concs[k][i,(j+1)%L] + concs[k][i,(j-1)%L]
                        -4*concs[k][i,j])
                        + q*concs[k][i,j]*(1-concs[0][i,j]-concs[1][i,j]-concs[2][i,j])
                        - p*concs[k][i,j]*concs[(k-1)%3][i,j])
    return newconcs

def type_field(concs):
    field = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            a = concs[0][i,j]
            b = concs[1][i,j]
            c = concs[2][i,j]
            r = 1-a-b-c
            els = [r,a,b,c]
            field[i,j] = np.argmax(els)
    return field

def animate(field, concs, nstep, d, q, p, dt):
    cmap = colors.ListedColormap(['gray', 'r', 'g', 'b'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure()
    plt.imshow(field, animated=True, cmap=cmap, norm=norm)
    plt.colorbar()

    for n in tqdm(range(nstep)):
        concs = update_concentrations(concs, d, q, p, dt)

        if n%50 == 0:
            field = type_field(concs)
            plt.cla()
            plt.title(n)
            plt.imshow(field, animated=True, cmap=cmap, norm=norm)
            plt.draw()
            plt.pause(0.001)
    plt.close()

#=======================================================
# Part a

def part_a():
    concentrations = [initial_state(L) for i in range(3)]
    field = type_field(concentrations)

    animate(field, concentrations, nstep=10000, d=1, q=1, p=0.5, dt=0.01)
#part_a()

#===========================================================

def get_fractions(field):
    ones = 0
    twos = 0
    threes = 0
    for e in field.flatten():
        if e == 1:
            ones += 1
        elif e == 2:
            twos += 1
        elif e == 3:
            threes += 1
    fracs = [ones/L**2, twos/L**2, threes/L**2]
    return fracs

def simulate_and_measure_fracs(concs, field, nstep, d, q, p):
    fracs = []
    times = []
    for n in tqdm(range(nstep)):
        concs = update_concentrations(concs, d=d, q=q, p=p, dt=0.01)

        if n%50==0:
            times.append(n)
            field = type_field(concs)
            fracs.append(get_fractions(field))
    return (times, fracs)

#===============================================================
# Part b

def part_b():
    concentrations = [initial_state(L) for i in range(3)]
    field = type_field(concentrations)

    times, fracs = simulate_and_measure_fracs(concentrations, field, nstep=20000, d=1, q=1, p=0.5)
    trans_frac = [[] for i in range(len(fracs[0]))]
    for line in fracs:
        for i, e in enumerate(line):
            trans_frac[i].append(e)

    singleax_multiplot_save(times, trans_frac,
                    '2022_exam/fractions_vs_time', 'Time', 'Fraction',
                    '', ['a', 'b', 'c'])
#part_b()

#====================================================================
    
def time_to_absorption(concs, field):
    iterations = 0
    while iterations*0.1 < 1000:
        concs = update_concentrations(concs, d=1, q=1, p=0.5, dt=0.1)
        field = type_field(concs)
        iterations += 1
        
        fracs = get_fractions(field)
        for frac in fracs:
            if 1 - frac < 0.1:
                return iterations
    return iterations

#=====================================================================
# Part c

def part_c():
    times = []
    for i in range(10):
        print(i)
        concentrations = [initial_state(L) for i in range(3)]
        field = type_field(concentrations)
        times.append(time_to_absorption(concentrations, field))
    times = [t for t in times if t*0.1 != 1000]
    print(f'Average time to absorption is {np.mean(times):.2f} and the error is {jacknife_error(times):.2f}.')
#part_c()

#======================================================================
# Part d

def part_d():
    concentrations = [initial_state(L) for i in range(3)]
    field = type_field(concentrations)

    animate(field, concentrations, nstep=10000, d=0.5, q=1, p=2.5, dt=0.01)

    concentrations = [initial_state(L) for i in range(3)]
    field = type_field(concentrations)

    times, fracs = simulate_and_measure_fracs(concentrations, field, nstep=20000, d=0.5, q=1, p=2.5)
    trans_frac = [[] for i in range(len(fracs[0]))]
    for line in fracs:
        for i, e in enumerate(line):
            trans_frac[i].append(e)

    singleax_multiplot_save(times, trans_frac,
                    '2022_exam/fractions_vs_time_part_d', 'Time', 'Fraction',
                    '', ['a', 'b', 'c'])
#part_d()

#======================================================================

def get_two_a_values(nstep):
    a1s = []
    a2s = []
    times = []
    concentrations = [initial_state(L) for i in range(3)]
    for n in tqdm(range(nstep)):
        concentrations = update_concentrations(concentrations, d=0.5, q=1, p=2.5, dt=0.01)
        
        a1s.append(concentrations[0][0,0])
        a2s.append(concentrations[0][30,30])
        times.append(n)
    return times, [a1s, a2s]

#=======================================================================
# Part e

def part_e():
    times, ajs = get_two_a_values(20000)
    singleax_multiplot_save(times, ajs,
                            '2022_exam/as_period', 'Iterations', 'Value of a',
                            '', ['a1', 'a2'])
#part_e()

#===========================================================================

def simulate_and_record_row_data(nstep, d):
    data = []
    concentrations = [initial_state(L) for i in range(3)]
    for n in tqdm(range(nstep)):
        concentrations = update_concentrations(concentrations, d=d, q=1, p=2.5, dt=0.01)
        field = type_field(concentrations)
        data.append(field[0, :int(L/2)])
    return data

def get_probability(data):
    probs = np.zeros(shape=len(data[0]))
    for line in data:
        for i in range(len(line)):
            if line[0] == line[i]:
                probs[i] += 1
    return probs/len(data)

#=========================================================================
# Part f

def part_f():
    ds = [0.5, 0.4, 0.3]
    datas = [simulate_and_record_row_data(3000, d) for d in ds]
    probs = [get_probability(data) for data in datas]
    dists = np.arange(len(probs[0]))
    xlabel='Distance from the first cell'
    ylabel='Probability of the same field on two cells'
    titles=[f'$D=${d}' for d in ds]
    filepathnames=['2022_exam/probability_vs_distance_' + str(int(10*d)) for d in ds]
    for i in range(3):
        scatter_and_save_data(dists, probs[i], filepathnames[i],
                            xlabel, ylabel, titles[i])
part_f()