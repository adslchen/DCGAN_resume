#!/usr/bin/env python3.4
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf

from model import DCGAN
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nIter', type=int, default=1000)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='completions')
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'Eye','Scarf'],
                    default='center')
parser.add_argument('--maskIter', type=int, default=0)
parser.add_argument('imgs', type=str, nargs='+')
parser.add_argument('--gpu',type=int, default=0)
parser.add_argument('--closeDisk',type=int, default=5)
parser.add_argument('--openDisk',type=int, default=4)
parser.add_argument('--threshold',type=float, default=0.85)
parser.add_argument('--blending',type=bool, default=False)
parser.add_argument('--loss',type=int, default=1)
args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.5
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    dcgan.complete(args)
