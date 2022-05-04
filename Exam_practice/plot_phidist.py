import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

f = open('Exam_practice/phi_vs_distance.dat')
data = f.readlines()
f.close()

phis = []
dists = []
for line in tqdm(data):
    i, j, phi = line.strip().split(' ')
    phis.append(float(phi))
    dists.append(np.sqrt((float(i)-25)**2 + (float(j)-25)**2))

plt.figure(figsize=(10,8))
plt.plot(dists, phis)
plt.xlabel('Distance from centre')
plt.ylabel('Phi value')
plt.savefig('Exam_practice/phi_vs_distance')
plt.show()