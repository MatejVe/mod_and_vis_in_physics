Code is split into two main files.
	1) game_of_life.py is an object oriented implementation of the game of life.
	   Tasks 1, 2 and 3 all implemented in that file. Associated plots and data 
	   files are task_2.png, equilibriate_times.dat (for equilibriation time),
	   and task_3_figure.png.
	2) SIRS model is implemented in the SIRS.py file. It is an object oriented
	   implementation as well. p1_p3_plane.py, p1_p3_var.py, cut_variance.py
	   and immunity.py are all python scripts that utilize the SIRS model.
	   They correspond to specific tasks within the checkpoint and have their
	   associated datafiles and plots.

I hope that the objects input parameters are clear just by their names, most of the
code is quite clean and should be readable. There is nothing unexpected in the code.

One 'mistake' I did make was I counted all 8 cells as nearest neighbours (so diagonal
cells were counted as well) in the SIRS model. This means that my plots are slightly 
different but I think the general features can still be seen. I didn't have time to 
fix this as I only realized only 4 nearest neighbours were meant to be used during 
the final workshop. I fixed this problem for the cut_with_error.png as otherwise the
plot is not useful at all. In addition, this plot was the least time consuming so I had
time to rerun it with appropriate number of neighbours. Rest of the plots will be a bit different
but qualitatively still the same. I hope this doesn't matter that much.
