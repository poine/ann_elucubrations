#! /usr/bin/env python
# -*- coding: utf-8 -*-


import math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

import utils as ut, aoa_utils

'''
   Plot the dataset, sequentially and as histograms
'''

def main():
    Vp, Ap, theta, airspeed = aoa_utils.read_dataset('../data/aoa_cleaned_2.csv')
    print('original dataset')
    aoa_utils.plot_sequential(Vp, Ap, theta, airspeed)
    aoa_utils.analyze_data(Vp, Ap)
    print('cooked dataset')
    Vp2, Ap2, theta2, airspeed2 = aoa_utils.cook_data(Vp, Ap, theta, airspeed)
    aoa_utils.analyze_data(Vp2, Ap2)
    plt.show()
      
if __name__ == "__main__":
    np.set_printoptions(linewidth=300)
    main()
