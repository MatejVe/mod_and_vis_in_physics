from SIRS import *

p1s = np.linspace(0, 1, 20)
p3s = np.linspace(0, 1, 20)

plotData = []

#f = open('p1_p3_plane.dat', 'w')
#for i, p1 in enumerate(p1s):
#    row = []
#    for j, p3 in enumerate(p3s):
#        print(f'p1-{i}/20, p3-{j}/20')
#        s = SIRS(50, p1, 0.5, p3)
#        infection = s.infectiveness_measure(1100, 100)
#        f.write(f'{p1} {p3} {infection}\n')
#        row.append(infection)
#    plotData.append(row)
#f.close()

f = open('p1_p3_plane.dat')
data = f.readlines()
plotData = [float(line.split(' ')[2]) for line in data]
plotData = np.array(plotData).reshape((20, 20))

X, Y = np.meshgrid(p1s, p3s)
plt.contourf(X, Y, plotData)

#plt.imshow(plotData, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.title('Average proportion of infected sites \n vs different spreading parameters, $p_2=0.5$')
plt.xlabel('$p_1$')
plt.ylabel('$p_3$')
plt.savefig('p1_p3_plane')
plt.close()