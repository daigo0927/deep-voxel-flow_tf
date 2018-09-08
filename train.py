import os
import time
import argparse
import numpy as np
import tensorflow as tf
from torch.utils import data

from datahandler.utils import get_dataset
from model import DeepVoxelFlow
from losses import L1loss
from utils import show_progress


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

        # L1 reproduction loss
        self.loss = L1loss(self.images[:,1], self.images_t_syn)
        # TODO: weight regularization (if required in the paper)
        # weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.model.vars])
        # self.loss = loss + self.args.lambda_*weights_l2

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
        for e in range(self.args.num_epochs):
            for i, (images, t) in enumerate(self.tloader):
                images = images.numpy()/255.0
                t = t.numpy()

                time_s = time.time()
                _, loss = self.sess.run([self.optimizer, self.loss],
                                        feed_dict = {self.images: images, self.t: t})

                if i%20 == 0:
                    batch_time = time.time() - time_s
                    kwargs = {'loss':loss, 'batch time'batch_time}
                    show_progress(e+1, i+1, self.num_batches, **kwargs)

            loss_vals = []
            for images_val, t_val in self.vloader:
                images_val = images_val.numpy()/255.0
                t_val = t_val.numpy()

                flow_val, loss_val \
                    = self.sess.run([self.flow, self.loss],
                                    feed_dict = {self.images: images_val, self.t: t_val})
                loss_vals.append(loss_val)

            print(f'\r{e+1} epoch validation, loss: {np.mean(loss_vals)}'\
                  +f', elapsed time: {time.time()-train_start} sec.')

            # TODO: visualize estimated results
            if self.args.visualize:
                if not os.path.exists('./figure'):
                    os.mkdir('./figure')
                

            if not os.path.exists('./model'):
                os.mkdir('./model')
            self.saver.save(self.sess, f'./model/model_{e+1}.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, required = True,
                        help = 'Target dataset, required')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset, required')
    parser.add_argument('--strides', type = int, default = 3,
                        help = 'Frame interval of training data [3]')
    parser.add_argument('--stretchable', action = 'store_true',
                        help = 'Stored option for frame sampling')
    
    parser.add_argument('--resolution', type = str, default = '480p',
                        choices = ['480p', 'Full-resolution'],
                        help = 'Resolution option for DAVIS dataset')
    parser.add_argument('--mode', type = str, default = 'clean',
                        choices = ['clean', 'final']
                        help = 'Image quality option for MPI-Sintel dataset')
    
    parser.add_argument('--num_epochs', type = int, default = 100,
                        help = '# of epochs [100]')
    parser.add_argument('--batch_size', type = int, default = 16,
                        help = 'Batch size [16]')
    parser.add_argument('--num_workers', type = int, default = 8,
                        help = '# of workers for data loading [8]')

    parser.add_argument('--crop_type', type = str, default = 'random',
                        help = 'Crop type for raw images [random]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [256, 320],
                        help = 'Crop shape for raw images [256, 320]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = None,
                        help = 'Resize shape for raw images [None]')
    parser.add_argument('--resize_scale', type = float, default = None,
                        help = 'Resize scale for raw images [None]')
    parser.add_argument('--image_size', nargs = 2, type = int, default = [256, 320],
                        help = 'Image size to be processed [256, 320]')

    parser.add_argument('--lr', type = float, default = 1e-4,
                        help = 'Learning rate [1e-4]')

    parser.add_argument('-v', '--visualize', dest = 'visualize', action = 'store_true',
                        help = 'Enable output visualization, [enabled]')
    parser.add_argument('--no_visualize', dest = 'visualize', action = 'store_false',
                        help = 'Disable output visualization, [enabled]')
    parser.add_argument(visualize = True)
    parser.add_argument('--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint file [None]')

    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f'{key} : {value}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
