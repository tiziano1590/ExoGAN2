import numpy as np
import os
class Grids():

    def __init__(self):
        wnw_file = os.path.join(os.path.dirname(__file__), 'wnw_grid.dat')
        self.wnw_grid = np.genfromtxt(wnw_file)