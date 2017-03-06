# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange
import numpy as np
from ops import *
from utils import *
from PIL import Image
from scipy import misc, io
from scipy.sparse import csr_matrix
from skimage.morphology import closing, opening, square, disk
import cv2

class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        # set for mask
        #self.batch_size = 1
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, 3]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                                    tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)



        self.X = tf.mul(1-self.mask, self.G) + tf.mul(self.mask, self.images)
        self.D_2, self.D_logits_2 = self.discriminator(self.X , reuse=True)
        self.perceptual_loss_2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_2,
                                                    tf.ones_like(self.D_2)))

        self.complete_loss_2 = self.contextual_loss + self.lam*self.perceptual_loss_2

        self.grad_complete_loss_2 = tf.gradients(self.complete_loss_2, self.z)



    def train(self, config):
        data = glob(os.path.join(config.dataset, "*.png"))
        #np.random.shuffle(data)
        assert(len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's Brandon Amos'
trained model for faces that's used in the post.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

""")
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

""")

        for epoch in xrange(config.epoch):
            data = glob(os.path.join(config.dataset, "*.png"))
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)



    def complete(self, config):
        #####################################################
        # This function was modified by Yu-An Chen and Wei-Che Chen
        #####################################################
        os.makedirs(os.path.join(config.outDir, 'hats_imgs'))
        os.makedirs(os.path.join(config.outDir, 'completed'))
        os.makedirs(os.path.join(config.outDir, 'mask'))
        os.makedirs(os.path.join(config.outDir, 'images'))
        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        #data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            scale = 0.25
            assert(scale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*scale)
            u = int(self.image_size*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        elif config.maskType == 'Eye':
            mask = np.ones(self.image_shape)
            mask[:26,:,:] = 0
        elif config.maskType == 'Scarf':
            mask = np.ones(self.image_shape)
            mask[25:,:,:] = 0
        else:
            assert(False)
        if config.maskIter is not 0:
            batch_idxs = config.maskIter     # if train for mask the batchidxs is the iteration number
            #config.imgs = ['data/lfw/test2/Tim_Robbins_0003.png']
            #img = get_image('data/lfw/test2/Tim_Robbins_0003.png', self.image_size, is_crop=self.is_crop)
            #image_shape = img.shape
            #img = np.resize(img,[1]+list(image_shape))
            masks = np.zeros((config.maskIter,self.image_shape[0],self.image_shape[1]))

        for idx in xrange(0, batch_idxs):
           # if config.maskIter is not 0:
           #     l = 0
           #     u = 1
           #     print "?"
          #  else:
          #  l = idx*self.batch_size
          #  u = min((idx+1)*self.batch_size, nImgs)
            l = 0
            u = 64
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = 8
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, batch_mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'masked{:02d}.png'.format(idx)))
            misc.imsave(os.path.join(config.outDir, 'origin.png'),255*inverse_transform(batch_images[21]))
            # implent the adam algorithm TODO
            #self.c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.999)\
            #                   .minimize(self.complete.loss)
            m = np.zeros((1,self.z_dim)).astype('float32')
            v = np.zeros((1,self.z_dim)).astype('float32')
            print("z_dim= ", self.z_dim)
            beta1 = float(0.9)
            beta2 = float(0.999)
            epil = 1e-8
            t = 0
            #print(self.grad_complete_loss)
            complete_x = np.zeros(batch_images.shape)
            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                #run = [self.complete_loss, self.grad_complete_loss, self.G]
                complete_loss = self.complete_loss_2
                grad_complete_loss = self.grad_complete_loss_2
                if config.loss == 0:
                    complete_loss = self.complete_loss
                    grad_complete_loss = self.grad_complete_loss
                run = [complete_loss, grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
                g = np.asarray(g[0])
                #print(g.shape)
                t += 1
                lr_t = config.lr *np.sqrt(1-beta2**t)/(1-beta1**t)
                m = beta1*m + (1-beta1)*g
                v = beta2*v + (1-beta2)*g**2
                #print("m_dim = ",m)
                zhats =  zhats -  config.lr*m/(np.sqrt(v)+epil)
                zhats = zhats.astype('float64')
                zhats = self.renorm(zhats, 1, 1)
                zhats = zhats.eval()
                #print(zhats.eval())
                #v_prev = np.copy(v)
                #v = config.momentum*v - config.lr*g[0]
                #zhats += -config.momentum * v_prev + (1+config.momentum)*v
                #zhats = np.clip(zhats, -1, 1)
                #mask = calc_mask(sel.)
                if i % 50 == 0 :
                    print("Val_loss=",i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    if config.maskIter is not 0:
                        imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:02d}.png'.format(idx))

                    nRows = np.ceil(batchSz/8)
                    nCols = 8
                    #if config.maskIter is not 0:
                    #misc.imsave(imgName,255*inverse_transform(G_imgs[21]))
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completeed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    if config.maskIter is not 0:
                        imgName = os.path.join(config.outDir,
                                           'completed/{:02d}.png'.format(idx))
                    #if config.maskIter is not 0:
                    #misc.imsave(imgName,255*inverse_transform(completeed[21]))

                    save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)
                if i == (config.nIter-1) :
                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    masked_images = np.multiply(batch_images, batch_mask)
                    complete_x = masked_images + inv_masked_hat_images
                    complete_x = inverse_transform(complete_x)

                    if idx == 0 and config.blending == True :
                        src = batch_images
                        dst = G_imgs
                        src = src[:,26:,:,:]
                        src = 255*inverse_transform(src)
                        dst = 255*inverse_transform(dst)
                        src = src.astype(np.uint8)
                        dst = dst.astype(np.uint8)

                        src_mask = 255*np.ones((src.shape[1],src.shape[2],src.shape[3]),src.dtype)

                        height = int(src.shape[1]/2)
                        width = int(src.shape[2]/2)

                        center = (width, height+26)

                        normal_clone = np.zeros(dst.shape)

                        for j in range(normal_clone.shape[0]):
                            normal_clone[j] = cv2.seamlessClone(src[j],dst[j],src_mask,center,cv2.NORMAL_CLONE)

                        imgName = os.path.join(config.outDir,'blending_{:02d}.png'.format(idx))
                        nRows = np.ceil(batchSz/8)
                        nCols = 8
                        imsave(normal_clone[:batchSz,:,:,:], [nRows,nCols],imgName)
                        complete_x = normal_clone
                    if idx == config.maskIter-1:
                   #     src = batch_images
                   #     dst = G_imgs
                   #     src = src.astype(np.float32)
                   #     dst = dst.astype(np.float32)
                   #     mask_1d = np.amax(mask, axis=3)


                   #     result = np.zeros(src.shape)

                    #    for j in range(src.shape[0]):
                    #        result[j] = self.blending(255*inverse_transform(src[j]),255*inverse_transform(dst[j]),mask_1d[j])
                        for j in range(complete_x.shape[0]):
                            imgName = os.path.join(config.outDir,'images/'+ batch_files[j][17:])
                            misc.imsave(imgName,complete_x[j])
                       # imgName = os.path.join(config.outDir,'blending_{:02d}.png'.format(idx))
                       # nRows = np.ceil(batchSz/8)
                       # nCols = 8
                       # imsave(result[:batchSz,:,:,:], [nRows,nCols],imgName)
                        #misc.imsave(imgName,result[21])






            if config.maskIter is not 0:
                print "?"
                print("================== update mask ==============================")
                mask = np.zeros(batch_mask.shape[:3])
                for ii in range(self.batch_size):
                    print ("calculating mask %d......"%(ii))
                    mask[ii] = self.calc_mask(255*inverse_transform(batch_images[ii]),
                                                255*inverse_transform(G_imgs[ii]),config)

                #mask = self.calc_mask(batch_images,G_imgs) ## Tim
                #masks[idx] = masks[idx]+mask                          # for merge the mask
                #misc.imsave(maskName,mask)
                #mask = np.reshape(mask, (mask.shape[0],mask.shape[1],mask.shape[2],1))
                #mask = np.repeat(mask, 3, axis=-1)

                maskMat = os.path.join(config.outDir,'mask/{:02d}.mat'.format(idx))
                io.savemat(maskMat,mdict = {'mask' : mask[4]})
                mask = np.reshape(mask, (batchSz, mask.shape[1], mask.shape[2],1))
                mask = np.repeat(mask, 3, axis=3)
                print "mask update ok !!!"
                maskName = os.path.join(config.outDir,'mask/{:02d}.png'.format(idx))
                nRows = np.ceil(batchSz/8)
                nCols = 8
                save_images(mask[:batchSz,:,:,:], [nRows,nCols], maskName)
                print mask.shape

        # These three line are for merge the mask
    #    masks = np.reshape(masks,(masks.shape[0],masks.shape[1],masks.shape[2],1))
    #    maskAname = os.path.join(config.outDir,'masksArray.png')
    #    save_images(masks,[8,8],maskAname)

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
            [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                        [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def renorm(self,x, axis, max_norm):
        
        #####################################################
        # This function was modified by Yu-An Chen and Wei-Che Chen
        # code reference to http://stackoverflow.com/questions/34934303/renormalize-weight-matrix-using-tensorflow
        #####################################################
        '''Renormalizes the sub-tensors along axis such that they do not exceed norm max_norm.'''
        #This elaborate dance avoids empty slices, which TF dislikes.
        rank = tf.rank(x)
        bigrange = tf.range(-1, rank + 1)
        dims = tf.slice(tf.concat(0, [tf.slice(bigrange,[0], [1 + axis]),
                        tf.slice(bigrange,[axis+2],[-1])]),[1],rank-[1])

        #Determine which columns need to be renormalized.
        l2norm_inv = tf.rsqrt(tf.reduce_sum(x*x, dims, keep_dims=True))
        scale = max_norm*tf.minimum(tf.cast(l2norm_inv,tf.float32), tf.constant(1.0/max_norm))

        #Broadcast the scalings
        return tf.mul(scale,x)

    def calc_mask(self,y0, x_recon0,config):
        #####################################################
        # This function was modified by Yu-An Chen and Wei-Che Chen
        #####################################################
        threshold = config.threshold
        median_e = 0.6
        mu_delta = 8
        w = np.zeros(y0.shape)

        for k in range(3):
            y = y0[:,:,k].flatten('F')
            y = np.reshape(y,(-1,1))
            y = np.array(y, dtype=np.float32)

            x_recon = x_recon0[:,:,k].flatten('F')
            x_recon = np.reshape(x_recon,(-1,1))
            x_recon = np.array(x_recon, dtype=np.float32)

            residual = np.square(y-x_recon)
            residual_sort = np.sort(residual, axis=0)
            delta = residual_sort[np.ceil(median_e*y.shape[0])]
            mu = (1.0*mu_delta)/(delta + 1e-10)
            wt = 1.0/(1+1.0/np.exp(-mu*(residual-delta)))
            w[:,:,k] = np.reshape(wt.T, (w.shape[1], w.shape[0])).T

        mk0 = (w[:,:,0] + w[:,:,1] + w[:,:,2])/3
        mk = mk0.flatten('F')
        mk[ np.argwhere(mk < threshold)] = 0
        mk[ np.argwhere(mk >= threshold)] = 1
        mk = np.reshape(mk.T, (y0.shape[1],y0.shape[0]))
        #for i in range(mk.shape[0]):
        #    mk[i] = closing(mk[i], disk(4))
        #    mk[i] = opening(mk[i], disk(2))
        mk = closing(mk, disk(config.closeDisk))
        mk = opening(mk, disk(config.openDisk))
        return mk.T

    def blending(self, source, target, mask):
        #####################################################
        # This function was modified by Yu-An Chen and Wei-Che Chen
        #####################################################

        temp1 = np.zeros((source.shape[0]+2,source.shape[1]+2,source.shape[2]))

        temp2 = np.zeros((source.shape[0]+2,source.shape[1]+2,source.shape[2]))

        for i in range(3):
            temp1[:,:,i] = np.lib.pad(source[:,:,i],(1,1),'symmetric')
            temp2[:,:,i] = np.lib.pad(target[:,:,i],(1,1),'symmetric')
        source = temp1
        target = temp2
        print("shape t", target.shape)
        mask = np.lib.pad(mask,(1,1),'constant', constant_values=(0,0))

        t_rows = target.shape[0]
        t_cols = target.shape[1]


        s = np.reshape(source,(t_rows*t_cols,-1),order='F')
        t = np.reshape(target,(t_rows*t_cols,-1),order='F')
        print("shape t ", t.shape)
        b = np.zeros((t_rows*t_cols, 3));
        print("shape b ",b.shape)
        print("constructing the matrix A...")
        print("shape s ", s.shape)

        #row_vec = np.zeros((t_rows*t_cols, 1))
        #col_vec = np.zeros((t_rows*t_cols,1 ))
        #value_vec = np.zeros((t_rows*t_cols,1 ))
        row_vec = []
        col_vec = []
        value_vec = []

        equation_num = 0


        for i in range(t_rows*t_cols):
            row = i%t_rows
            col = i/t_rows
            if mask[row,col] != 0:
                b[i,:] = 4*s[i,:] - s[i-1,:] - s[i+1,:] - s[i+t_rows,:] - s[i-t_rows,:]
                #b[i,:] = temp[ii]
                row_vec.append(i)
                col_vec.append(i)
                value_vec.append(4)
                equation_num += 1
                #equation_num = equation_num%(t_rows*t_cols)

                row_vec.append(i)
                col_vec.append(i+1)
                value_vec.append(-1)
                equation_num += 1
                #equation_num = equation_num%(t_rows*t_cols)

                row_vec.append(i)
                col_vec.append(i-1)
                value_vec.append(-1)
                equation_num += 1
                #equation_num = equation_num%(t_rows*t_cols)

                row_vec.append(i)
                col_vec.append(i-t_rows)
                value_vec.append(-1)
                equation_num += 1
                #equation_num = equation_num%(t_rows*t_cols)

                row_vec.append(i)
                col_vec.append(i+t_rows)
                value_vec.append(-1)
                equation_num += 1
                #equation_num = equation_num%(t_rows*t_cols)
            else:
                row_vec.append(i)
                col_vec.append(i)
                value_vec.append(1)
                equation_num += 1
                #equation_num = equation_num%(t_rows*t_cols)

                b[i,:] = t[i,:]
        row_vec = np.asarray(row_vec)
        col_vec = np.asarray(col_vec)
        value_vec = np.asarray(value_vec)
        print("row_vec",row_vec.shape)
        print("col_vec",col_vec.shape)
        #print(value)
        #A = (row_vec, col_vec, value_vec, t_rows*t_cols, t_rows*t_cols)
        A = csr_matrix((value_vec, (row_vec, col_vec)), shape=(t_rows*t_cols, t_rows*t_cols)).toarray()

        f_red = np.linalg.solve(A,b[:,0])
        f_green = np.linalg.solve(A,b[:,1])
        f_blue = np.linalg.solve(A,b[:,2])

        f_red = np.reshape(f_red, (t_rows, t_cols), order = 'F')
        f_green = np.reshape(f_green, (t_rows, t_cols), order ='F')
        f_blue = np.reshape(f_blue, (t_rows, t_cols), order ='F')

        result = np.zeros((t_rows, t_cols, 3))
        result[:,:,0] = f_red
        result[:,:,1] = f_green
        result[:,:,2] = f_blue

        result = result[1:t_rows-1, 1:t_cols-1, :]
        return result


