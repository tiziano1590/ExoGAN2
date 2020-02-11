import numpy as np

class Grids():

    def __init__(self):
        super().__init__()
        self.wnw_grid = np.genfromtxt("./wnw_grid.dat")