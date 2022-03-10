from SIRS import *

plt.style.use('dark_background')

p1s = np.linspace(0.2, 0.5, 20)

plotData = []
errors = []

f = open('cut_variance.dat','w')
for i, p1 in enumerate(p1s):
    print(f'p1-{i}/20')
    s = SIRS(50, p1, 0.5, 0.5)
    var, err = s.infectiveness_variance_measure_with_error(10000, 100, 5)
    f.write(f'{p1} {var} {err}\n')
    plotData.append(var)
    errors.append(err)
f.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.errorbar(p1s, plotData, yerr=errors, fmt='o')
ax.set_title('Variance in the infection ratio vs $p_1$ ($p_2 = 0.5$, $p_3=0.5$)')
ax.set_xlabel('Value of $p_1$')
ax.set_ylabel('Variance')
plt.savefig('cut_with_error')
plt.close()