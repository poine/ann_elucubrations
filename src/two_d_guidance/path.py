import numpy as np, scipy.signal


import pdb

class Path:

    def __init__(self, **kwargs):
        self.last_passed_idx = 0 # we don't allow going back
        if 'load' in kwargs:
            self.load(kwargs['load'])
        elif 'points' in kwargs:
            self.initialize(kwargs['points'], kwargs.get('headings', []), kwargs.get('dists', []))
        else:
            self.clear()

    def __str__(self):
        return 'points {} headings {} curvatures {} dists {}'.format(self.points, self.headings, self.curvatures, self.dists)

    def initialize(self, points, headings=[], dists=[]):
        self.points = points
        if len(headings) > 0:
            self.headings = headings
        else:
            self.compute_headings()
        if len(dists) > 0:
            self.dists = dists
        else:
            self.compute_dists()

    def clear(self):
        self.points = np.empty((0, 2))
        self.headings = np.empty((0))
        self.curvatures = np.empty((0))
        self.dists = np.empty((0))

    def load(self, filename):
        print('loading path from {}'.format(filename))
        data =  np.load(filename)
        #pdb.set_trace()
        # self.points = np.array([p.tolist() for p in data['points']]) # why???
        self.points = data['points']
        try:
            self.headings = data['headings']
        except KeyError:
            print(' -no headings in archive, computing them')
            self.compute_headings()
        try:
            self.dists = data['dists']
        except KeyError:
            print(' -no dists in archive, computing them')
            self.compute_dists()
        try:
            self.curvatures = data['curvatures']
        except KeyError:
            print(' -no curvatures in archive, computing them')
            self.compute_curvatures()

    def save(self, filename):
        print('saving path to {}'.format(filename))
        np.savez(filename, points=self.points, headings=self.headings, curvatures=self.curvatures, dists=self.dists)

    def compute_headings(self):
        self.headings = np.zeros(len(self.points))
        for i, p in enumerate(self.points):
            if i==0: dp = self.points[1]-self.points[0]
            elif i==len(self.points)-1: dp = self.points[-1]-self.points[-2]
            else: dp = self.points[i+1]-self.points[i-1]
            self.headings[i] = np.arctan2(dp[1], dp[0])

    def compute_dists(self):
        self.dists = np.zeros(len(self.points))
        for i, p in enumerate(self.points[1:]):
            self.dists[i+1] = self.dists[i] + np.linalg.norm(self.points[i+1]-self.points[i])

    def compute_curvatures(self):
        self.curvatures = np.zeros(len(self.points))
        # http://www.ambrsoft.com/TrigoCalc/Circle3D.htm
        for i in range(1, len(self.points) -1):
            (x1, y1), (x2, y2), (x3, y3) = self.points[i-1], self.points[i] , self.points[i+1]
            d1, d2, d3 = x1**2+y1**2, x2**2+y2**2, x3**2+y3**2
            A = np.linalg.det(np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]))
            B = np.linalg.det(np.array([[d1, y1, 1], [d2, y2, 1], [d3, y3, 1]]))
            C = np.linalg.det(np.array([[d1, x1, 1], [d2, x2, 1], [d3, x3, 1]]))
            D = -np.linalg.det(np.array([[d1, x1, y1], [d2, x2,  y2], [d3, x3, y3]]))
            if abs(A)<1e-9:
                self.curvatures[i] = 0.
            else:
                R2 = (B**2 + C**2 - 4*A*D) / 4 / A**2
                R = np.sqrt(R2)
                self.curvatures[i] = 1/R
                v1, v2 = self.points[i]-self.points[i-1], self.points[i+1]-self.points[i-1]
                cross = v1[0]*v2[1]-v1[1]*v2[0]
                self.curvatures[i] *= np.sign(cross)
        self.curvatures[0] = self.curvatures[1]
        self.curvatures[-1] = self.curvatures[-2]
        # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
        #self.inv_radius_filtered = scipy.signal.savgol_filter(self.inv_radius, 13, 1, mode='nearest', deriv=0)
        #v0, alpha = 0.9, 0.075
        #self.vel_sp = v0*np.exp(-alpha*self.inv_radius_filtered)


    def move_point(self, i, p, y):
        self.points[i] = p
        self.headings[i] = y
        if i > 0:
            self.dists[i] = self.dists[i-1] + np.linalg.norm(self.points[i]-self.points[i-1])

    def insert_points(self, i, points, headings, dists, curvatures):
        self.points = np.insert(self.points, i, points, axis=0)
        self.headings = np.insert(self.headings, i, headings)
        if dists is None: # compute straight line distances
            dists = np.insert(np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)), [0], 0)
        self.dists = np.insert(self.dists, i, (self.dists[-1] if len(self.dists) else 0) + dists)
        self.curvatures = np.insert(self.curvatures, i, curvatures)

    def append_points(self, points, headings, dists=None, curvatures=None):
        self.insert_points(len(self.points), points, headings, dists, curvatures)

    def append(self, others):
        for other in others:
            self.append_points(other.points, other.headings, other.dists, other.curvatures)


    def is_finished(self): return self.last_passed_idx == len(self.points) - 1

    def reset(self):
        #print('reset')
        self.last_passed_idx = 0

    # def enter(self, p0):
    #     i = np.argmin(np.linalg.norm(p0 - self.points, axis=1))
    #     self.last_passed_idx = i
    #     return i
        
    def find_closest(self, p0, max_look_ahead=100):
        ''' find point on path closest to p0 '''
        i = np.argmin(np.linalg.norm(p0 - self.points[self.last_passed_idx:self.last_passed_idx+max_look_ahead], axis=1)) + self.last_passed_idx
        self.last_passed_idx = i
        return i, self.points[i]


    def find_closest_looped(self, p0, max_look_ahead=100):
        i, p1 = self.find_closest(p0, max_look_ahead)
        end_reached = False
        if i == len(self.points)-1 and np.linalg.norm(p0-self.points[0]) <= np.linalg.norm(p0-self.points[-1]):
            i = 0
            end_reached = True; self.last_passed_idx = 0; print('end reached')
        return i, self.points[i], end_reached


    def find_point_at_dist_from_idx(self, i, _d=0.2):
        j=i
        while j<len(self.points) and self.dists[j] - self.dists[i] < _d:
            j += 1
        return (j, self.points[j]) if j < len(self.points) else (None, None)


    def find_point_at_dist_from_idx_looped(self, i, _d):
        j=i
        while j<len(self.points) and self.dists[j] - self.dists[i] < _d:
            j += 1
        if j == len(self.points):
            j = 0; _d-=(self.dists[-1]-self.dists[i])
            while self.dists[j] < _d:
                j += 1
        return j
    
    def find_carrot_alt(self, p0, _d=0.2): # FIXME: change that name
        i1, p1 = self.find_closest(p0)
        i2, p2 = self.find_point_at_dist_from_idx(i1, _d)
        end_reached = (i2 is None)
        return p1, p2, end_reached, i1, i2
 
    def find_carrot_looped(self, p0, _d):
        ''' returns the closest path point to p0 as well as a point on path ahead at distance _d '''
        i, p1, end_reached = self.find_closest_looped(p0)
        j = self.find_point_at_dist_from_idx_looped(i, _d)
        #_np = len(self.points)
        #print(f"{p0} {i}/{_np} {j}")
        return p1, self.points[j], end_reached, i, j

    def find_carrots_looped(self, p0, _ds):
        i1, p1, end_reached = self.find_closest_looped(p0)
        _is = [i1]
        for d in _ds:
            _is.append(self.find_point_at_dist_from_idx_looped(_is[-1], d))
        return _is, end_reached
            
    def get_extends(self):
        xmin, ymin = np.amin(self.points, axis=0)
        xmax, ymax = np.amax(self.points, axis=0)
        #return xmin, ymin, xmax, ymax
        x_center, y_center = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
        dx, dy = xmax-xmin, ymax-ymin
        return np.array((x_center, y_center)), np.array((dx, dy))
