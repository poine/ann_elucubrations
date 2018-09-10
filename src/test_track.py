#!/usr/bin/env python
import logging, sys, math, numpy as np
import matplotlib.pyplot as plt

import two_d_guidance as tdg


def test():
    #_t = tdg.track_factory.make_circle_track([0, 0], 0.4, 0.7, 0, 2*np.pi, 360)
    _t = tdg.track_factory.make_oval_track([-1, 0], [1, 0], 0.5, 0.3)
    _t = tdg.track_factory.make_fig_of_height_track(0.5, 0.3)
    tdg.track_factory.view(_t)
    plt.show()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test()
    
