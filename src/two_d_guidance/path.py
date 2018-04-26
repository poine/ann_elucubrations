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
        return 'points {} headings {} dists {}'.format(self.points, self.headings, self.dists)

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
        self.dists = np.empty((0))
        self.curvatures = np.empty((0))

    def load(self, filename):
        print('loading from {}'.format(filename))
        data =  np.load(filename)
        self.points = np.array([p.tolist() for p in data['points']])
        self.headings = data['headings']
        if 'dists' in data:
            self.dists = data['dists']
        else:
            self.compute_dists()
        self.compute_curvatures()

    def save(self, filename):
        print('saving to {}'.format(filename))
        np.savez(filename, points=self.points, headings=self.headings, dists=self.dists)

    def reset(self):
        print 'reset'
        self.last_passed_idx = 0

    def compute_headings(self):  # TEST ME
        self.headings = np.zeros(len(self.points))
        for i, p in enumerate(self.points):
            self.headings[i] = numpy.arctan2(self.points[i+1]-self.points[i])

    def compute_dists(self):
        self.dists = np.zeros(len(self.points))
        for i, p in enumerate(self.points[1:]):
            self.dists[i+1] = self.dists[i] + np.linalg.norm(self.points[i+1]-self.points[i])

    def compute_curvatures(self):
        self.radius= np.zeros(len(self.points))
        self.inv_radius= np.zeros(len(self.points))
        # http://www.ambrsoft.com/TrigoCalc/Circle3D.htm
        for i in range(2, len(self.points) -1):
            (x1, y1), (x2, y2), (x3, y3) = self.points[i-1], self.points[i] , self.points[i+1]
            d1, d2, d3 = x1**2+y1**2, x2**2+y2**2, x3**2+y3**2
            A = np.linalg.det(np.array([[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]))
            B = np.linalg.det(np.array([[d1, y1, 1], [d2, y2, 1], [d3, y3, 1]]))
            C = np.linalg.det(np.array([[d1, x1, 1], [d2, x2, 1], [d3, x3, 1]]))
            D = -np.linalg.det(np.array([[d1, x1, y1], [d2, x2,  y2], [d3, x3, y3]]))
            if abs(A)<1e-9:
                self.radius[i] = float('inf')
                self.inv_radius[i] = 0.
            else:
                R2 = (B**2 + C**2 - 4*A*D) / 4 / A**2
                R = np.sqrt(R2)
                self.radius[i] = R
                self.inv_radius[i] = 1/R
        self.radius[0] = self.radius[1]
        self.radius[-1] = self.radius[-2]
        self.inv_radius[0] = self.inv_radius[1]
        self.inv_radius[-1] = self.inv_radius[-2]
        # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
        self.inv_radius_filtered = scipy.signal.savgol_filter(self.inv_radius, 13, 1, mode='nearest', deriv=0)
        v0, alpha = 0.9, 0.075
        self.vel_sp = v0*np.exp(-alpha*self.inv_radius_filtered)


    def move_point(self, i, p, y):
        self.points[i] = p
        self.headings[i] = y
        if i > 0:
            self.dists[i] = self.dists[i-1] + np.linalg.norm(self.points[i]-self.points[i-1])

    def insert_points(self, i, p, y, d=None):
        self.points = np.insert(self.points, i, p, axis=0)
        self.headings = np.insert(self.headings, i, y)
        if d is None: # compute straight line distances
            d = np.insert(np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1)), [0], 0)
        self.dists = np.insert(self.dists, i, (self.dists[-1] if len(self.dists) else 0) + d)

    def append_points(self, p, y, d=None):
        self.insert_points(len(self.points), p, y, d)

    def append(self, others):
        for other in others:
            self.append_points(other.points, other.headings, other.dists)


    def find_closest(self, p0, max_look_ahead=100):
        i = np.argmin(np.linalg.norm(p0 - self.points[self.last_passed_idx:self.last_passed_idx+max_look_ahead], axis=1)) + self.last_passed_idx
        self.last_passed_idx = i
        return i, self.points[i]


    def find_carrot(self, i, p1, _d=0.2):
        j=i
        while j<len(self.points) and self.dists[j] - self.dists[i] < _d:
            j += 1
        return (j, self.points[j]) if j < len(self.points) else (None, None)

    def find_carrot_alt(self, p0, _d=0.2):
        i, p1 = self.find_closest(p0)
        j, p2 = self.find_carrot(i, p1, _d)
        end_reached = (j is None)
        return p1, p2, end_reached, i, j


    def find_entry_point(self, p0):
        return
