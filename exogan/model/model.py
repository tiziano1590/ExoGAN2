# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
# import keras
import math
import numpy as np
import itertools
from glob import glob
import pickle
import gzip
import tensorflow as tf
import matplotlib.pyplot as plt
from six.moves import xrange
import sys

sys.path.append('./libraries')

from libraries.ops import *
from libraries.library_grid import *
from util import *

class DCGAN(object):
    def __init__(self, sess, image_size=72, is_crop=False,
                 batch_size=64, sample_size=64, lowres=8,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1,
                 checkpoint_dir=None, lam=0.1, data_chunks=200,
                 kernel_size=21):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            lowres: (optional) Low resolution image/mask shrink factor. [8]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [1]
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        # assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]
        self.kernel_size = kernel_size

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = c_dim
        self.data_chunks = data_chunks

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i, )) for i in range(4)]

        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i, )) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        # self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
        #     [self.batch_size, self.lowres_size, self.lowres,
        #      self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        # self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
        #     [self.batch_size, self.lowres_size, self.lowres,
        #      self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)

        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, trainpars):
        
        epoch =        int(trainpars['epoch'])
        learning_rate = float(trainpars['learning_rate'])
        beta1 =         float(trainpars['beta1'])
        train_size =    float(trainpars['train_size'])
        if train_size == 1e99:
            train_size =      np.inf
        batch_size =    int(trainpars['batch_size'])
        image_size =    int(trainpars['image_size'])
        dataset =       str(trainpars['dataset'])
        checkpoint_dir = str(trainpars['checkpoint_dir'])
        sample_dir =    str(trainpars['sample_dir'])
        log_dir =       str(trainpars['log_dir'])
        
        dataset_file = glob.glob(os.path.join(dataset, "*.{}".format("pgz")))
        np.random.shuffle(dataset_file)
        data_chunks = list(chunks(dataset_file, self.data_chunks))
        np.random.shuffle(data_chunks)

        assert (len(dataset_file) > 0)

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))
        # sample = data[0:self.sample_size]
        # print(sample.__sizeof__())

        # sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        # sample_images = np.array(sample).astype(np.float32)
        # print(sample_images.size)
        # print(sample_images[0])

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""
======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's a model for faces
trained on the CelebA dataset for 20 epochs.
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

        for epoch in xrange(epoch):
            for data_train in data_chunks:
                data = data_array(data_train)
                batch_idxs = min(len(data), train_size) // self.batch_size

                for idx in xrange(0, batch_idxs):
                    batch = data[idx * batch_size:(idx + 1) * batch_size]
                    nanmean = np.nanmean(batch)
                    batch[np.where(np.isnan(batch) == True)] = nanmean
                    # sys.exit()

                    # batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                    #          for batch_file in batch]
                    # print(batch.__sizeof__())
                    batch_images = np.array(batch).astype(np.float32)
                    batch_images = batch_images.reshape((self.batch_size, self.image_size, self.image_size, self.c_dim))
                    # print(batch_images.size)
                    # sys.exit()

                    batch_z = np.random.uniform(-1, 1, [batch_size, self.z_dim]).astype(np.float32)

                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.images: batch_images, self.z: batch_z,
                                                              self.is_training: True})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.is_training: True})
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.z: batch_z, self.is_training: True})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                    errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                    counter += 1
                    print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                    if np.mod(counter, 100) == 1:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.G, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: batch_images, self.is_training: False}
                        )
                        save_images(samples, [8, 8],
                                    './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                    if np.mod(counter, 500) == 2:
                        self.save(checkpoint_dir, counter)

    def complete(self, comppars, test_single=''):
        
        outDir = comppars['outDir']
        maskType = comppars['maskType']
        centerScale = comppars['centerScale']
        nIter = int(comppars['nIter'])
        outInterval = int(comppars['outInterval'])
        approach = comppars['approach']
        
        beta1 = comppars['beta1']
        beta2 = comppars['beta2']
        lr = comppars['lr']
        eps = comppars['eps']
        
        hmcL = int(comppars['hmcL'])
        hmcEps = comppars['hmcEps']
        hmcBeta = comppars['hmcBeta']
        hmcAnneal = comppars['hmcAnneal']
        
        planet = comppars['synth_planets']
        planet_prob = comppars['synth_planet_probability']
        test_directory = comppars['save_synth_planets_out_info']
        if test_directory[-1] != '/': test_directory += '/'
        
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('hats_imgs')
        make_dir('histograms')
        make_dir('completed')
        make_dir('logs')
        make_dir(test_directory)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert (isLoaded)

        nImgs = 1

        batch_idxs = int(np.ceil(nImgs / self.batch_size))
        # lowres_mask = np.zeros(self.lowres_shape)
        if maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif maskType == 'center':
            assert (centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size * centerScale)
            u = int(self.image_size * (1.0 - centerScale))
            mask[l:u, l:u, :] = 0.0
        elif maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:, :c, :] = 0.0
        elif maskType == 'full':
            mask = np.ones(self.image_shape)
        elif maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4, ::4, :] = 1.0
        elif maskType == "parameters":
            mask = np.ones(self.image_shape)
            mask[61:, :] = 0.0
            mask[:, 61:] = 0.0
        else:
            assert (False)
        
        batchSz = self.batch_size
            
        for idx in xrange(0, batch_idxs):
            batch = test_image_from_fits(test_single,
                                         comppars['tup'],
                                         planet=comppars['planet'],
                                         batch_size=self.batch_size,
                                         imgsz=self.image_size, 
                                         kernel_size=self.kernel_size)
            batch = batch.reshape(self.batch_size, self.image_size, self.image_size)
            batch_images = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
            batch_images[:batchSz, :, :, 0] = batch
            batch_images = np.array(batch_images).astype(np.float32)
            if batchSz < self.batch_size:
                padSz = ((0, int(self.batch_size - batchSz)), (0, 0), (0, 0), (0, 0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz, :, :, :], [nRows, nCols],
                        os.path.join(outDir, 'before.png'))
            masked_images = np.multiply(batch_images, mask)
            save_images(masked_images[:batchSz, :, :, :], [nRows, nCols],
                        os.path.join(outDir, 'masked.png'))

            for i in xrange(nIter):
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.images: batch_images,
                    self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)
                deciphered = decipher_lpa(G_imgs)
                try:
                    truth_img = np.load(test_single)
                    truth_img = truth_img.reshape(1, 72, 72, 1)
                    deciphered["input_file"] = test_single
                    deciphered["truths"] = {}
                    deciphered["truths"] = decipher_lpa(truth_img)
                except (ValueError, OSError):
                    pass


                if i % outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    pickle.dump(deciphered, open(str(os.path.join(outDir, "histograms/output.pickle")), "wb"))

                    nRows = np.ceil(batchSz / 8)
                    nCols = min(8, batchSz)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)

                    completed = masked_images + inv_masked_hat_images

                if approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = beta1 * m_prev + (1 - beta1) * g[0]
                    v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - beta1 ** (i + 1))
                    v_hat = v / (1 - beta2 ** (i + 1))
                    zhats += - np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                    zhats = np.clip(zhats, -1, 1)

                elif approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, self.z_dim)
                    v_old = np.copy(v)

                    for steps in range(hmcL):
                        v -= hmcEps / 2 * hmcBeta * g[0]
                        zhats += hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= hmcEps / 2 * hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = hmcBeta * loss_old[img] + np.sum(v_old[img] ** 2) / 2
                        logprob = hmcBeta * loss[img] + np.sum(v[img] ** 2) / 2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    hmcBeta *= hmcAnneal

                else:
                    assert (False)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim * 2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim * 4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim * 8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')

            # print(tf.nn.sigmoid(h4))

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * 5 * 5, 'g_h0_lin', with_w=True)

            # TODO: Nicer iteration pattern here. #readability
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 5, 5, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1  # Iteration number.
            depth_mul = 8  # Depth decreases as spatial component increases.
            size = 9  # Size increases as depth decreases.

            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i - 1],
                                               [self.batch_size, size, size, self.gf_dim * depth_mul], name=name,
                                               with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1],
                                           [self.batch_size, size, size, self.c_dim], name=name, with_w=True)
            # print(self.batch_size, size, size, self.c_dim)
            # tf.print(tf.nn.tanh(hs[i])[0, :, :, 0])
            # sys.exit()

            return tf.nn.sigmoid(hs[i])

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
