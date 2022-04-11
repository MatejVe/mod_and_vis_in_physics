Cahn-Hilliard equation solver and all the plots related to it are stored in the folder CH.
There are two main scripts:
	CH.py which uses numpy and np.roll to calculate the derivatives. Two main functions are used
	animate() which runs an animation and free_energy which plots the free energy for a lattice.
	It can be run as a script but I didn't put in the option of inputing the parameters through the command pannel.

	CH_loops.py is equivalent to CH.py but uses for loops to update the lattice, efficiency is achieved through the usage
	of the numba package.

	Data is stored in the buble_fe.dat file (free energy data for uneven densities of the two fluids)
	and uniform_fe.dat (free energy data for even densities of the two fluids).

Poisson equation solvers and all the plots related to it are stored in the folder Poisson.

	1. jacobi_algorithm uses the jacobi update rule to solve the equation, it stores the data in
	center_charge.dat (single charge in the middle of the box) and center_charge_distance.dat (potential
	vs absolute distance from the centered charge). Plots are included. They are plotted in the plotter_electric.py file.
	jacoby_algorithm.py can be run from the CMD with arguments N atol chargeDistribution.

	2. electric_GS is the same as the jacobi_algorithm but uses the Gauss-Seidel algorithm. Data is saved
	to the same files.

	3. GS_over_relax calculates and plots the best value of w for the overrelaxation method. Tolerance which is
	considered for convergence can be changed within the script.

	4. magnetic_GS.py solves the Poisson equation for the magnetic potential of a single wire within the box.
	The data is stored in the one_current.dat file.

	5. Field plots are stored in the efield_arrow_cutz50.png and bfield_arrows.png

	6. 3D_magnetic_electric_py is a script that creates 3D plots of the electric and magnetic fields. Plots
	are saved into the 3D_electric_field.png and 3D_magnetic_field.png