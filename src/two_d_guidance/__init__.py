import math

from path import *
from path_factory import *
from pure_pursuit import *


def norm_yaw(y):
    while y >= math.pi:
        y -= 2*math.pi
    while y < -math.pi:
        y += 2*math.pi
    return y



class VelController:
    def __init__(self, v_sp=0.75):
        self.K = -2.
        self.v_sp = v_sp

    def compute(self, v):
        return self.K*(v-self.v_sp)
