import numpy as np
from . import path

class Track:
    def __init__(self, **kwargs):
        if 'load' in kwargs:
            self.load(kwargs['load'])
        else:
            self.clear()

    def clear(self):
        self.left_border =  np.empty((0, 2))
        self.right_border =  np.empty((0, 2))
        self.path = path.Path()


    def load(self, filename):
        print('loading track from {}'.format(filename))
