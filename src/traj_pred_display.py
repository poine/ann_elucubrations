#! /usr/bin/env python
# -*- coding: utf-8 -*-
import logging, sys, time, dateutil, gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

import numpy as np, matplotlib.pyplot as plt

import utils as ut
import test_traj_pred as tpu
from test_traj_pred import Trajectory
#import traj_pred_ann_tflearn_4 as tpi






class Plot(Gtk.Frame):
    def __init__(self):
        Gtk.Frame.__init__(self)
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.add(self.canvas)
        self.axis = figure.add_subplot(111)
        self.axis.set_aspect('equal')
        figure.subplots_adjust(bottom=0., top=1., left=0., right=1.)
        self.set_size_request(640, 480)
        self.marker = None
        
    def update_trajectory(self, traj):
        self.axis.clear()
        self.axis.plot(traj.pos[:,0], traj.pos[:,1])
        self.canvas.draw()

    def update_marker(self, traj, t_marker):
        i_marker = np.where(traj.time >= t_marker)[0][0]
        #print t_marker, i_marker, traj.time[i_marker]
        if self.marker is not None:
            self.axis.lines.remove(self.marker)
        self.marker = self.axis.plot(traj.pos[i_marker,0], traj.pos[i_marker,1], marker='o', color='green')[0]
        self.canvas.draw()

class PredictionPlot(Gtk.Frame):
    def __init__(self):
        Gtk.Frame.__init__(self)
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.add(self.canvas)
        self.axis = figure.add_subplot(111)
        self.axis.set_aspect('equal')
        figure.subplots_adjust(bottom=0., top=1., left=0., right=1.)
        self.set_size_request(640, 480)

        
class View:
    def __init__(self):
        self.b = Gtk.Builder()
        self.b.add_from_file("traj_viewer.xml")
        self.window = self.b.get_object("window")
        self.plot = Plot()
        self.b.get_object("alignment_plot").add(self.plot)
        self.pred_plot = PredictionPlot()
        self.b.get_object("alignment_pred_plot").add(self.pred_plot)
        self.window.show_all()

    def load_traj(self, traj):
        s = self.b.get_object("scale_marker") 
        s.set_adjustment(Gtk.Adjustment(traj.time[0], traj.time[0], traj.time[-1], 5, 10, 0))

        self.b.get_object('label_start').set_text('{}'.format(traj.time[0]))
        self.b.get_object('label_end').set_text('{}'.format(traj.time[-1]))
        self.plot.update_trajectory(traj)
        
    def update_marker(self, traj, t_marker):
        self.plot.update_marker(traj, t_marker)
        

class App:
    def __init__(self, ds_filename):
        self.view = View()
        self.ds = tpu.DataSet(load=ds_filename)
        self.traj_id = 8
        self.setup_view()
 
    def setup_view(self):
        self.view.window.connect("delete-event", self.quit)
        self.view.load_traj(self.ds.trajectories[self.traj_id])
        self.view.b.get_object("scale_marker").connect("value-changed", self.on_set_marker)

    def on_set_marker(self, s):
        t_marker = float(s.get_value())
        self.view.update_marker(self.ds.trajectories[self.traj_id], t_marker)
        
    def run(self):
        Gtk.main()        

    def quit(self, a, b):
        Gtk.main_quit()


def plot_2d_pred(ds, traj_id):
    
    ut.prepare_fig(fig=None, window_title=None, figsize=(10.24, 10.24), margins=None)
    plt.plot(ds.trajectories[traj_id].pos[:,0], ds.trajectories[traj_id].pos[:,1])
    plt.axes().set_aspect('equal')
    ut.decorate(plt.gca(), title='Two D view', xlab='East (m)', ylab='North (m)', legend=None, xlim=None, ylim=None)
        
def main():
    nps, nvs, horiz, ils = 1, 1, 10, []
    a = tpi.AnnAgent(nps=nps, nvs=nvs, horiz=horiz, ils=ils)
    a.load('/tmp/traj_pred_tfl4_1_1_10_forced')
    a.report()

    d = tpu.DataSet(load='../data/bdx_20130914_25ft.pkl')
    t_id = 8
    plot_2d_pred(d, t_id)
    #tpu.test_predictions(a, d.trajectories, [horiz])
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    App('../data/bdx_20130914_25ft.pkl').run()
    #main()
