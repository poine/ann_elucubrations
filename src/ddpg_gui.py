#! /usr/bin/env python
# -*- coding: utf-8 -*-

#
# This is a graphical interface for running DDPG agents
#


#import dql__gym_pendulum as dql # don't know why this has to go first...
import ddpg_agent
import tensorflow as tf
import gym, gym_foo
import os, argparse, logging, sys, time, threading, dateutil, gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, GObject
import numpy as np

import pdb

class View:
    def __init__(self):
        self.b = Gtk.Builder()
        self.b.add_from_file("ddpg_gui.xml")
        self.window = self.b.get_object("window")
        self.window.show_all()


    def request_path(self, action):
        dialog = Gtk.FileChooserDialog("Please choose a directory", self.window, action,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        ret = dialog.run()
        file_path = dialog.get_filename() if ret == Gtk.ResponseType.OK else None
        dialog.destroy()
        return file_path


    def set_busy(self, what):
        if what == 'train':
            self.b.get_object('button_train').set_label('Pause')
        for act in ['load', 'save', 'train', 'test']:
            if act != what:
                self.b.get_object('button_{}'.format(act)).set_sensitive(False)
        self.timeout_id = GLib.timeout_add(100, self.on_timeout, None)

    def set_not_busy(self, what):
        if what == 'train':
            self.b.get_object('button_train').set_label('Start')
        for act in ['load', 'save', 'train', 'test']:
            self.b.get_object('button_{}'.format(act)).set_sensitive(True)
        GLib.source_remove(self.timeout_id)
        
    def on_timeout(self, user_data):
        self.b.get_object('progressbar_activity').pulse()
        return True

        
class App:
    def __init__(self):
        self.view = View()
        config = ddpg_agent.Config()
        parser = config.setup_cmd_line_parser()
        args = config.parse_cmd_line(parser)
        #config.dump()
        self.model = ddpg_agent.Model(config)
        self.setup_view()

        self.gym_thread_event = threading.Event()
        self.gym_thread = threading.Thread(name='gym_thread', target=self.__gym_thread_main)
        self.gym_thread_quit = False
        self.gym_thread_work = None
        self.gym_thread.start()

    def setup_view(self):
        self.view.window.connect('delete-event', self.quit)
        for act in ['load', 'save', 'train', 'test']:
            self.view.b.get_object('button_{}'.format(act)).connect('clicked', self.on_button_clicked, act)

        c = self.model.config.cfg
        for prm, keys, fmt, parse_fn in [('nb_episodes', ['max_episodes'], '{:d}', int),
                                         ('exploration_noise', ['agent', 'actor', 'noise_sigma'], '{:f}', float)]:
            self.view.b.get_object('entry_{}'.format(prm)).connect('activate', self.on_param_changed, (keys, fmt, parse_fn))
            self.view.b.get_object('entry_{}'.format(prm)).set_text(fmt.format(self.model.config.get(keys)))

        self.view.b.get_object('check_render_env').connect('toggled', self.on_render_toggled)
        self.view.b.get_object('label_env_name').set_text(c['env']['name'])
        self.view.b.get_object('check_render_env').set_active(c['render_env'])

    def run(self):
        #self.model.load_agent('/home/poine/work/ann_elucubrations/src/saves/bicycle_18')
        Gtk.main()        

    def quit(self, a, b):
        self.gym_thread_quit = True
        self.gym_thread_event.set()
        self.gym_thread.join()
        self.model.__exit__(None, None, None)
        Gtk.main_quit()

    def on_button_clicked(self, button, action):
        print(action)
        if action == 'load':
            dirname = self.view.request_path(Gtk.FileChooserAction.SELECT_FOLDER)#Gtk.FileChooserAction.OPEN)
            if dirname is not None:
                self.model.load_agent(dirname)
        elif action == 'save':
            dirname = self.view.request_path(Gtk.FileChooserAction.SELECT_FOLDER)#Gtk.FileChooserAction.SAVE)
            if dirname is not None:
                self.model.save_agent(dirname)
        elif action == 'test':
            #self.view.set_busy('test')
            self.gym_thread_work = 'test'
            self.gym_thread_event.set()
        elif action == 'train':
            if not self.model.is_training:
                #self.view.set_busy('train')
                self.gym_thread_work = 'train'
                self.gym_thread_event.set()
            else:
                self.model.abort_training()

    def on_param_changed(self, entry, args):
        keys, fmt, parse_fn = args
        print(keys, fmt, parse_fn)
        try:
            val = parse_fn(entry.get_text())
            self.model.config.set(keys, val)
        except ValueError:
            entry.set_text(fmt.format(self.model.config.get(keys)))

    def on_render_toggled(self, button):
        self.model.config.cfg['render_env'] = button.get_active()


    def __gym_thread_main(self):
        while not self.gym_thread_quit:
            print('#### gym_tread sleeping')
            event = self.gym_thread_event.wait()
            print('#### gym_tread awaken {}'.format(event))
            self.gym_thread_event.clear()
            if self.gym_thread_work == 'test':
                self.model.test_agent()
            elif self.gym_thread_work == 'train':
                self.model.run_training()
            self.gym_thread_work = None
                
                
    # def __train(self):
    #     def body():
    #         print('## starting training')
    #         self.model.run_training()
    #         print('## training done')
    #         self.view.set_not_busy('train')
    #         self.view.b.get_object('label_episode').set_text('{}'.format(self.model.agent.episode_nb))
    #     #if self.model.agent.episode_nb == 0:
    #     #    self.model.init_training()
    #     self.view.set_busy('train')
    #     t = threading.Thread(target=body)
    #     t.start()
        
            
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(linewidth=300)
    App().run()
