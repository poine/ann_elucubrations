import numpy as np


class Param:
    def __init__(self):
        self.M = 1    # mass of the quad
        self.m = 0.5  # mass of the pole
        self.L = 0.5  # half width of the quand
        self.l = 0.5  # dist to pole cg
        self.J = 0.01 # inertia of the quad
        self.j = 0.01 # inertia of the pole

    def compute_aux(self):
        self.mt = self.M + self.m
        self.mr = self.m / self.mt
        self.LovJ = self.L/self.J


class PVTP:
    s_x, s_z, s_th, s_ph, s_xd, s_zd, s_thd, s_phd, s_size = range(9) # state vector components
    i_f1, i_f2, i_size =  range(3) # input vector components
    
    def __init__(self, P=None):
        self.P = P if P is not None else Param()

    
    def dyn(self, X, t, U):
        sth, cth = np.sin(X[PVTP.s_th]), np.cos(X[PVTP.s_th])
        sph, cph = np.sin(X[PVTP.s_ph]), np.cos(X[PVTP.s_ph])
        phdsq = X[PVTP.s_ph]*X[PVTP.s_ph] # phi_dot_squared
        ut =  U[PVTP.i_f1] + U[PVTP.i_f2]
        ud = -U[PVTP.i_f1] + U[PVTP.i_f2]

        thdd = self.P.LovJ*ud # quad angular accel
        
        
        Xd = np.zeros(8)
        return Xd
