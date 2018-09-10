import math

from two_d_guidance.path import *
from two_d_guidance.path_factory import *
from two_d_guidance.pure_pursuit import *
from two_d_guidance.track import *
from two_d_guidance.track_factory import *

class VelController:
    def __init__(self, v_sp=0.75):
        self.K = -2.
        self.v_sp = v_sp

    def compute(self, v):
        return self.K*(v-self.v_sp)



class PurePursuitVelController:
    def __init__(self, path_file, params, look_ahead=0.3, v_sp=0.5):
        self.ppc = PurePursuit(path_file, params, look_ahead)
        self.vc = VelController(v_sp)

    def compute(self, cur_pos, cur_psi, cur_vel):
        #try:
        U = np.array(self.ppc.compute(cur_pos, cur_psi))
        #except pure_pursuit.EndOfPathException:
        #    self.ppc.path.reset()
        #    U = [0,0]
        U[0] = self.vc.compute(cur_vel)
        return U
