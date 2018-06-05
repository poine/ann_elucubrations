

import math, numpy as np

def norm_angle(a):
    while a>math.pi: a-=2*math.pi
    while a<-math.pi: a+=2*math.pi
    return a

class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        
    def run(self, dt, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])




class PVTOL:
    def __init__(self):
        
        self.gravity = 9.8
        self.mass = 0.5
        self.l = 0.2
        self.J = 0.01

    def run(self, dt, f1, f2):
        x, z, th, xd, zd, thd = self.state
        cth, sth = np.cos(th), np.sin(th)
        ut, ud = f1+f2, f2-f1

        xdd = -sth/self.mass*ut
        zdd =  cth/self.mass*ut - self.gravity
        thdd = self.l/self.J*ud

        xd  += xdd*dt
        zd  += zdd*dt
        thd += thdd*dt

        x += xd*dt
        z += zd*dt
        th += thd*dt
        self.state = np.array([x, z, norm_angle(th), xd, zd, thd])
        return self.state

    def jac(self):
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, -self.gravity, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [0, 0],
                      [1/self.mass, 1/self.mass],
                      [-self.l/self.J, self.l/self.J]])
        return A, B

        
#
#  Linear reference models
#

class LinRef:
    ''' Linear Reference Model (with first order integration)'''
    def __init__(self, K):
        '''K: coefficients of the caracteristic polynomial, in ascending powers order,
              highest order ommited (normalized to -1)'''
        self.K = K; self.order = len(K)
        self.X = np.zeros(self.order+1)

    def run(self, dt, sp):
        self.X[:self.order] += self.X[1:self.order+1]*dt
        e =  np.array(self.X[:self.order]); e[0] -= sp
        self.X[self.order] = np.sum(e*self.K)
        return self.X

    def poles(self):
        return np.roots(np.insert(np.array(self.K[::-1]), 0, -1))

    def reset(self, X0=None):
        if X0 is None: X0 = np.zeros(self.order+1)
        self.X = X0


class FirstOrdLinRef(LinRef):
    def __init__(self, tau):
        LinRef.__init__(self, [-1/tau])

class SecOrdLinRef(LinRef):
    def __init__(self, omega, xi):
        LinRef.__init__(self, [-omega**2, -2*xi*omega])

def make_random_pulses(dt, size, min_nperiod=1, max_nperiod=10, min_intensity=-1, max_intensity=1.):
    ''' make a vector of pulses of random duration and intensities '''
    npulses = size/max_nperiod*2
    durations = np.random.random_integers(low=min_nperiod, high=max_nperiod, size=npulses)
    intensities =  np.random.uniform(low=min_intensity, high=max_intensity, size=npulses)
    pulses = []
    for duration, intensitie in zip(durations, intensities):
        pulses += [intensitie for i in range(duration)]
    pulses = np.array(pulses)
    time = np.linspace(0, dt*len(pulses), len(pulses))
    return time, pulses


    


"""
Compute numerical jacobian 
"""
def num_jacobian(X, U, dyn):
    s_size = len(X)
    i_size = len(U)
    epsilonX = (0.1*np.ones(s_size)).tolist()
    dX = np.diag(epsilonX)
    A = np.zeros((s_size, s_size))
    for i in range(0, s_size):
        dx = dX[i,:]
        delta_f = dyn(X+dx/2, 0, U) - dyn(X-dx/2, 0, U)
        delta_f = delta_f / dx[i]
        A[:,i] = delta_f

    epsilonU = (0.1*np.ones(i_size)).tolist()
    dU = np.diag(epsilonU)
    B = np.zeros((s_size,i_size))
    for i in range(0, i_size):
        du = dU[i,:]
        delta_f = dyn(X, 0, U+du/2) - dyn(X, 0, U-du/2)
        delta_f = delta_f / du[i]
        B[:,i] = delta_f

    return A,B

