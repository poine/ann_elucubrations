#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time, numpy as np
import gym_foo.envs.pvtol_pole as pvtp


if __name__ == '__main__':
    np.set_printoptions(linewidth=300, suppress=True)
    v = pvtp.PVTP()
    
    X, U = np.zeros(8), np.zeros(2)
    Xd = v.dyn(X, 0, U)
    print Xd
