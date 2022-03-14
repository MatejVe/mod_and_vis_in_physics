from SIRS import *

plt.style.use('dark_background')

imunProp = np.linspace(0, 1, 20)

plotData = []
errors = []

f = open('immunity.dat', 'w')
for i, prop in enumerate(imunProp):
    print(f'imun-{i}/{len(imunProp)}')
    freeze = []
    for j in range(5):
        s = SIRS(50, 0.5, 0.5, 0.5, immune=prop)
        inf = s.infectiveness_measure(1000, 100)
        freeze.append(inf)
    avg = np.mean(freeze)
    err = np.std(freeze)
    f.write(f'{prop} {avg} {err}\n')
    plotData.append(avg)
    errors.append(err)
f.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.errorbar(imunProp, plotData, yerr=errors, fmt='o')
ax.set_title('Average infection vs immune ratio')
ax.set_xlabel('Immune ratio')
ax.set_ylabel('Average infectio ratio')
plt.savefig('immunity')
plt.close()