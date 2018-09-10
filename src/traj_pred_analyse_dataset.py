#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, matplotlib.pyplot as plt
import pdb


import utils as ut
import test_traj_pred as tpu
from test_traj_pred import Trajectory

def main():
    d = tpu.DataSet(load='../data/bdx_20130914_25ft.pkl')

    traj = d.trajectories[8]
    
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
