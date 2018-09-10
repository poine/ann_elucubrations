
import math, numpy as np


class Dynamics:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.reset()
        
    def reset(self, X0=np.zeros(4)):
        self.state = np.array(X0)
        return self.state
        
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
        return self.state


from gym.envs.classic_control import rendering

class Rendering:
    def __init__(self, screen_width = 600, screen_height = 400, world_width=4.):
        self.screen_width = screen_width
        self.scale = screen_width/world_width
        self.carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = self.scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        self.viewer = rendering.Viewer(screen_width, screen_height)
        l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        axleoffset =cartheight/4.0
        cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        self.carttrans = rendering.Transform()
        cart.add_attr(self.carttrans)
        self.viewer.add_geom(cart)
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        pole.set_color(.8,.6,.4)
        self.poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(self.poletrans)
        pole.add_attr(self.carttrans)
        self.viewer.add_geom(pole)
        self.axle = rendering.make_circle(polewidth/2)
        self.axle.add_attr(self.poletrans)
        self.axle.add_attr(self.carttrans)
        self.axle.set_color(.5,.5,.8)
        self.viewer.add_geom(self.axle)
        self.track = rendering.Line((0,self.carty), (screen_width,self.carty))
        self.track.set_color(0,0,0)
        self.viewer.add_geom(self.track)

    def render(self, x, theta, mode):
        cartx = x*self.scale+self.screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, self.carty)
        self.poletrans.set_rotation(-theta)
        
        return self.viewer.render(return_rgb_array = (mode=='rgb_array'))
