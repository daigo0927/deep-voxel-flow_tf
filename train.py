import os
import time
import argparse
import numpy as np
import tensorflow as tf
from torch.utils import data

from datahandler.utils import get_dataset
from model import DeepVoxelFlow


class Trainer:
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        dset = get_dataset(self.args.dataset)
        data_args = {'dataset_dir':self.args.dataset_dir,
                     'strides':self.args.strides, 'stretchable':self.args.stretchable,
                     'cropper':self.crop_type, 'crop_shape':self.args.crop_shape,
                     'resize_shape':self.args.resize_shape, 'resize_scale':self.args.resize_scale}
        if self.args.dataset == 'davis_v':
            data_args['resolution'] = self.args.resolution
        elif self.args.dataset == 'Sintel_v':
            data_args['mode'] = self.args.mode

        tset = dset(train_or_val = 'train', **data_args)
        vset = dset(train_or_val = 'val', **data_args)

        load_args = {'batch_size':self.args.batch_size, 'num_workers':self.args.num_workers,
                     'pin_memory':True, 'drop_last':True}
        self.num_batches = int(len(self.tset)/self.args.batch_size)
        self.tloader = data.DataLoader(tset, shuffle = True, **load_args)
        self.vloader = data.DataLoader(vset, shuffle = False, **load_args)

    def _build_graph(self):
        self.images = tf.placeholder(tf.float32, shape = [None, 3]+self.ars.image_size+[3]
                                     name = 'images')
        self.t = tf.placeholder(tf.flaot32, shape = [None], 't')

        self.model = DeepVoxelFlow(name = 'dvf')
        self.images_t_syn, self.flow = self.model(self.images[:,0], self.images[:,-1], self.t)

        # TODO: loss implementation
        loss = loss_func(self.images[:,1], self.images_t_syn)
        # TODO: weight regularization (if required in the paper)
        weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.model.vars])
        self.loss = loss + self.args.lambda_*weights_l2

        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.args.lr)\
                         .minimize(self.loss, var_list = self.model.vars)
        self.saver = tf.train.Saver()

        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())

    def train(self):
        train_start = time.time()
        
