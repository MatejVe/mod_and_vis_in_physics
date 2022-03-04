import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

b = [0, 1, 2]
a = [0, 1, 2]
c = [4, 5, 6]
d = [7, 8, 9]
e = [10, 11, 12]

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes[0, 0].scatter(b, a)
axes[0, 0].set_xlabel('Abcssd')
axes[0, 1].scatter(b, c)
axes[1, 0].scatter(b, d)
axes[1, 1].scatter(b, e)
plt.tight_layout()
plt.show()