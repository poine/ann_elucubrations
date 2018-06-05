import numpy as np, scipy.integrate
import misc

'''

Dynamic model for the PVTOL pole system.
See https://poine.github.io/ann_elucubrations/pvtol_pole.html for equations

'''


class Param:
    def __init__(self):
        self.M = 1    # mass of the quad
        self.m = 0.5  # mass of the pole
        self.L = 0.5  # half width of the quand
        self.l = 0.5  # dist to pole cg
        self.J = 0.01 # inertia of the quad
        self.j = 0.01 # inertia of the pole
        self.g = 9.81
        self.compute_aux()
        
    def compute_aux(self):
        self.mt = self.M + self.m
        self.oneovmt = 1./self.mt
        self.mr = self.m / self.mt
        self.lmr = self.l*self.mr
        self.LovJ = self.L/self.J
        self.lmovjpml2 = self.l*self.m/(self.j+self.m*self.l*self.l)

class PVTP:
    s_x, s_z, s_th, s_ph, s_xd, s_zd, s_thd, s_phd, s_size = range(9) # state vector components
    i_f1, i_f2, i_size =  range(3) # input vector components
    
    def __init__(self, P=None):
        self.P = P if P is not None else Param()
        self.Ue = self.P.mt*self.P.g/2*np.ones(2)
    
    def dyn(self, X, t, U):
        sth, cth = np.sin(X[PVTP.s_th]), np.cos(X[PVTP.s_th])
        sph, cph = np.sin(X[PVTP.s_ph]), np.cos(X[PVTP.s_ph])
        phdsq = X[PVTP.s_ph]*X[PVTP.s_ph] # phi_dot_squared
        ut =  U[PVTP.i_f1] + U[PVTP.i_f2]
        ud = -U[PVTP.i_f1] + U[PVTP.i_f2]

        thdd = self.P.LovJ*ud # quad angular accel
        
        a = -self.P.lmr*cph
        e = -self.P.lmr*sph*phdsq -self.P.oneovmt*sth*ut
        b = -self.P.lmr*sph
        f =  self.P.lmr*cph*phdsq - self.P.g + self.P.oneovmt*cth*ut
        c = -self.P.lmovjpml2*cph
        d = -self.P.lmovjpml2*sph
        h =  self.P.lmovjpml2*sph*self.P.g
        A = np.array([[1, 0, a],[0, 1, b],[c, d, 1]])
        Y = np.array([[e],[f],[h]])
        xdd, zdd, phdd = np.dot(np.linalg.inv(A), Y)
        Xd = np.zeros(8)
        Xd[:4] = X[4:]
        Xd[4:] = [xdd, zdd, thdd, phdd]
        return Xd

    def jac(self):
        Xe = np.zeros(8)
        return misc.num_jacobian(Xe, self.Ue, self.dyn)

    def disc_dyn(self, Xk, Uk, dt):
        _unused, Xkp1 = scipy.integrate.odeint(self.dyn, Xk, [0, dt], args=(Uk,))
        return Xkp1
