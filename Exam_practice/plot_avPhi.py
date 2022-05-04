import matplotlib.pyplot as plt
from tqdm import tqdm

f = open('Exam_practice/av_phi.dat')

data = f.readlines()

f.close()

phi = []
iter = []

for line in tqdm(data):
    p, i = line.strip().split(' ')
    phi.append(float(p))
    iter.append(int(i))

plt.figure(figsize=(10,8))
plt.plot(iter, phi)
plt.xlabel('Iterations')
plt.ylabel('Average phi value')
plt.savefig('Exam_practice/average_vs_iterations')
plt.show()