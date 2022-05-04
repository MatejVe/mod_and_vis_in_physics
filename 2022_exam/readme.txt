--------------------------------------

Part b

There is a long period of only a mixed phase in the beggining. After a certain number of iterations,
individual phases begin to show up. They quickly change and oscillate until one phases absorbes the whole lattice
and the system gets into an absorbing state.

Related plots: part_b_snapshot.png, fractions_vs_time.png
-------------------------------------

Part c

Time to absorption is: 2035 iterations
Standard deviation in that is 1040 iterations
(dt = 0.1)
---------------------------------------

Part d

There is no period of only mixed phases. Each phase pops up quite quickly and they oscilate in
approximately even proportions. There is no sign of tending towards an absorbing state. From the fractions plot
we can see that the system enters an oscillating phase where all three phases 'fight' for dominance
but neither succeeds.

Related plots: part_d_snapshot.png, fractions_vs_time_part_d.png
---------------------------------------

Part e

After some time to go through equlibriation it can be seen that the a concentration of two random lattice points
oscillates with approximately equal frequency.

Related plots: as_period.png
--------------------------------------

Part f

Probability of two cells being in the same phase vs the distance between them seems to fall of more quickly as the value
of the parameter D is increased. For D=0.3, the probability falls of and then starts slightly increasing near the furthests
cells. For D=0.4, the same happens but there is an observed peak and then the probability starts to fall off again.
For D=0.5, we can observe two peaks and the probability falls much more rapidly with increasing distance.


--------------------------------------
--------------------------------------
Notes:

All the code is contained withing the concentrations.py file, which is split in parts (just like in the exam sheet)
and the auxiliary_functions.py file which are functions I have prepared in advance to make my exam easier.
Most of the functions in there are unused and I am particularly sad I didn't have time to adapt some of the more
advanced functions on this specific exam because of the akwardness of calculating the concentrations and the resulting field
separately.

I hope the rest of the code is clear and somewhat easily understandable.