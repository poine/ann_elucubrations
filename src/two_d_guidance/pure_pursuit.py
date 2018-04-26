
import sys , math, numpy as np

import path

class EndOfPathException(Exception):
    pass

class PurePursuit:
    def __init__(self, path_file, params, look_ahead=0.3):
        self.path = path.Path(load=path_file)
        self.params = params
        self.look_ahead = look_ahead

    def compute(self, cur_pos, cur_psi):
        p1, p2, end_reached, ip1, ip2 = self.path.find_carrot_alt(cur_pos, _d=self.look_ahead)
        if end_reached:
            raise EndOfPathException

        p0p2_w = p2 - cur_pos
        cy, sy = math.cos(cur_psi), math.sin(cur_psi)
        w2b = np.array([[cy, sy],[-sy, cy]])
        p0p2_b = np.dot(w2b, p0p2_w)
        l = np.linalg.norm(p0p2_w)
        R = (l**2)/(2*p0p2_b[1])
        return 0, math.atan(self.params.L/R)
        #return R, p2 # Radius and carrot 
