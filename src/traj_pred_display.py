#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, logging, sys, time, dateutil, gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas

import numpy as np, matplotlib.pyplot as plt
import pdb


import utils as ut
import test_traj_pred as tpu
from test_traj_pred import Trajectory
import traj_pred_ann_tflearn_4 as tpi
from traj_pred_ann_tflearn_4 import Param


class Plot(Gtk.Frame):
    def __init__(self):
        Gtk.Frame.__init__(self)
        figure = Figure()
        self.canvas = FigureCanvas(figure)
        self.add(self.canvas)
        self.axis = figure.add_subplot(111)
        self.axis.set_aspect('equal')
        figure.subplots_adjust(bottom=0., top=1., left=0., right=1.)
        


class TrajPlot(Plot):
    def __init__(self):
        Plot.__init__(self)
        self.set_size_request(480, 480)
        self.marker = None
        
    def update_trajectory(self, traj):
        self.axis.clear()
        self.axis.plot(traj.pos[:,0], traj.pos[:,1])
        self.canvas.draw()

    def update_marker(self, traj, i_marker):
        try:
            self.axis.lines.remove(self.marker)
        except ValueError: pass
        self.marker = self.axis.plot(traj.pos[i_marker,0], traj.pos[i_marker,1], marker='o', color='green')[0]
        self.canvas.draw()
    
        
class PredictionPlot(Plot):
    def __init__(self):
        Plot.__init__(self)
        self.set_size_request(640, 320)
 
    def update_prediction(self, traj, i_marker, pred):
        self.axis.clear()
        _s, _e = max(0, i_marker-10), i_marker+len(pred)+2 if pred is not None else  i_marker + 10
        self.axis.plot(traj.pos[_s:_e,0], traj.pos[_s:_e,1])
        if pred is not None: self.axis.plot(pred[:,0], pred[:,1])
        self.axis.plot(traj.pos[i_marker,0], traj.pos[i_marker,1], marker='o', color='green')[0]
        self.canvas.draw()


class View:
    def __init__(self):
        self.b = Gtk.Builder()
        self.b.add_from_file("traj_viewer.xml")
        self.window = self.b.get_object("window")
        self.traj_plot = TrajPlot()
        self.b.get_object("alignment_plot").add(self.traj_plot)
        self.pred_plot = PredictionPlot()
        self.b.get_object("alignment_pred_plot").add(self.pred_plot)
        self.window.show_all()

    def load_agent(self, a):
        tv = self.b.get_object("tv_agent")
        tv.get_buffer().set_text(a.report())
        
    def load_dataset(self, ds):
        l = self.b.get_object("label_nb_traj")
        l.set_text('{}'.format(len(ds.trajectories)))
        sp = self.b.get_object("sp_traj")
        sp.set_adjustment(Gtk.Adjustment(0, 0, len(ds.trajectories), 1, 5, 0))
        
    def load_traj(self, traj):
        s = self.b.get_object("scale_marker") 
        s.set_adjustment(Gtk.Adjustment(traj.time[0], traj.time[0], traj.time[-1], 1, 5, 0))
        s.set_value(traj.time[0])
        self.b.get_object('label_start').set_text('{}'.format(traj.time[0]))
        self.b.get_object('label_end').set_text('{}'.format(traj.time[-1]))
        self.traj_plot.update_trajectory(traj)
        
    def update_marker(self, traj, i_marker, pred):
        self.traj_plot.update_marker(traj, i_marker)
        self.pred_plot.update_prediction(traj, i_marker, pred)
        

class App:
    def __init__(self, ds_filename, ag_filename):
        self.view = View()
        self.ds = tpu.DataSet(load=ds_filename)
        self.a = tpi.AnnAgent(load=ag_filename)
        self.traj_id = 8 #wtf!!!
        self.setup_view()
 
    def setup_view(self):
        self.view.window.connect("delete-event", self.quit)
        self.view.load_agent(self.a)
        self.view.load_dataset(self.ds)
        self.view.load_traj(self.ds.trajectories[self.traj_id])
        self.view.b.get_object("scale_marker").connect("value-changed", self.on_set_marker)
        self.view.b.get_object("sp_traj").connect('value-changed', self.on_traj_changed)

    def on_traj_changed(self, e):
        try:
            self.traj_id = int(e.get_text())
            _traj = self.ds.trajectories[self.traj_id]
            self.view.load_traj(_traj)
            self.view.update_marker(_traj, self.a.delay, None)
        except ValueError:
            e.set_text('{}'.format(self.traj_id))
        
    def on_set_marker(self, s):
        t_marker = float(s.get_value())
        traj = self.ds.trajectories[self.traj_id]
        i_marker = np.where(traj.time >= t_marker)[0][0]
        pred = self.a.predict2(traj, i_marker) if i_marker >= self.a.delay else None
        self.view.update_marker(traj, i_marker, pred)

        
    def run(self):
        Gtk.main()        

    def quit(self, a, b):
        Gtk.main_quit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset', default='../data/bdx_20130914_25ft.pkl')
    parser.add_argument('--agent',   help='agent', default='../data/traj_pred/agent_1/a')
    args = vars(parser.parse_args())
    App(args['dataset'], args['agent']).run()
    
