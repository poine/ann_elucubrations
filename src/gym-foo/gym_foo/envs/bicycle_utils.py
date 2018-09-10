import os, numpy as np
from gym.envs.classic_control import rendering
import pdb

class BicycleViewer:

    def __init__(self, screen_width, screen_height, cockpit_height, paths, carrots_nb, cfg):
        
        self.viewer = rendering.Viewer(screen_width, screen_height)
        scene_with, scene_height = screen_width, screen_height# - cockpit_height

        xmin, xmax, ymin, ymax = np.PINF, np.NINF, np.PINF, np.NINF 
        for p in paths:
            w_center, w_extends = p.get_extends()
            _xmin, _xmax = w_center[0]-w_extends[0]/2, w_center[0]+w_extends[0]/2
            _ymin, _ymax = w_center[1]-w_extends[1]/2, w_center[1]+w_extends[1]/2
            xmin, xmax = np.min([xmin, _xmin]), np.max([xmax, _xmax])
            ymin, ymax = np.min([ymin, _ymin]), np.max([ymax, _ymax])
        w_extends, w_center = [xmax-xmin, ymax-ymin], [(xmin+xmax)/2, (ymin+ymax)/2]
        scale = min(scene_with/w_extends[0], scene_height/w_extends[1])*0.7
        world_width, world_height = scene_with/scale, scene_height/scale
        xmin, xmax = w_center[0] - world_width/2,  w_center[0]+world_width/2
        ymin, ymax = w_center[1] - world_height/2, w_center[1]+world_height/2
        self.viewer.set_bounds(xmin, xmax, ymin, ymax) #left, right, bottom, top
        for p in paths:
            self.viewer.add_geom(rendering.PolyLine(p.points, True))#, color=obj.color2, linewidth=2)

        carwidth, carlen = cfg['car_w'], cfg['car_l']#0.1, 0.2
        l,r,t,b = -carlen/2, carlen/2, carwidth/2, -carwidth/2
        car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
        self.cartrans = rendering.Transform()
        car.add_attr(self.cartrans)
        self.viewer.add_geom(car)

        # self.carrot_transs = [rendering.Transform() for i in range(carrots_nb+1)]
        # for _t in self.carrot_transs:
        #     _c = rendering.make_circle(0.04)
        #     _c.set_color(.4,.8,.6)
        #     _c.add_attr(_t)
        #     self.viewer.add_geom(_c)
        
        self.carrot_rel_transs = [rendering.Transform() for i in range(carrots_nb+1)]
        for _t in self.carrot_rel_transs:
            _c = rendering.make_circle(carwidth/3)
            _c.set_color(.8,.6,.4)
            _c.add_attr(_t)
            _c.add_attr(self.cartrans)
            self.viewer.add_geom(_c)

            
        wheelx, wheely =  0.4*world_width, 0.32*world_height
        img_fname = os.path.join(os.path.dirname(__file__), "assets/steering_wheel.png")
        print(scale)
        wheel_img = rendering.Image(img_fname, 70./scale, 70./scale)
        self.wheel_img_transf = rendering.Transform(translation=(wheelx, wheely))
        wheel_img.add_attr(self.wheel_img_transf)
        self.viewer.add_geom(wheel_img)


        speedo_x, speedo_y = -0.4*world_width, 0.32*world_height
        img_fname = os.path.join(os.path.dirname(__file__), "assets/speedometer.png")
        speedo_img = rendering.Image(img_fname, 100./scale, 100./scale)
        speedo_img_transf = rendering.Transform(translation=(speedo_x, speedo_y))
        speedo_img.add_attr(speedo_img_transf)
        self.viewer.add_geom(speedo_img)
        
        speedo_needle = rendering.FilledPolygon([(-2,0), (0,40), (2,0)])
        speedo_needle.set_color(.8,.6,.4)
        self.speedo_transf = rendering.Transform(translation=(speedo_x, speedo_y), scale=(1/scale,1/scale))
        speedo_needle.add_attr(self.speedo_transf)
        self.viewer.add_geom(speedo_needle)
        
        self.throttle_x, self.throttle_y, self.throttle_dy =  0.4*world_width, 0.1*world_height, -0.001*world_height
        img_fname = os.path.join(os.path.dirname(__file__), "assets/pedal_green.png")
        acc_pedal_img = rendering.Image(img_fname, 70./scale, 70./scale)
        acc_pedal_transf = rendering.Transform(translation=(self.throttle_x, self.throttle_y))
        acc_pedal_img.add_attr(acc_pedal_transf)
        self.viewer.add_geom(acc_pedal_img)

        throttle_needle = rendering.FilledPolygon([(-4,0), (0,15), (4,0)])
        throttle_needle.set_color(.8,.6,.4)
        self.throttle_transf = rendering.Transform(translation=(self.throttle_x, self.throttle_dy), scale=(1/scale,1/scale))
        throttle_needle.add_attr(self.throttle_transf)
        self.viewer.add_geom(throttle_needle)
        
    
    def render(self, x, y, psi, v, steering, throttle, carrots_b, mode):
        self.cartrans.set_translation(x, y)
        self.cartrans.set_rotation(psi)

        #for c, tc in zip(self.carrots_w, self.carrot_transs):
        #    tc.set_translation(c[0], c[1])

        for c, tc in zip(carrots_b, self.carrot_rel_transs):
            tc.set_translation(c[0], c[1])
            
        self.wheel_img_transf.set_rotation(2*steering)
        self.throttle_transf.set_translation(self.throttle_x-0.2*throttle, self.throttle_dy)
        self.speedo_transf.set_rotation(-v+np.pi/2)
         
        res = self.viewer.render(return_rgb_array = mode=='rgb_array')
        return res
