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
from scipy.stats import chisquare
import sys
import copy

from exogan.parameter import ParameterParser
from exogan.util import *


class DCGAN(object):
    def __init__(self, sess, genpars):
        """
        Args:
            sess: TensorFlow session
            genpars: Dictionary of general parameter to define the DCGAN network
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        ## assert (image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.is_crop = bool(genpars["is_crop"])
        self.batch_size = int(genpars["batch_size"])
        self.image_size = int(genpars["image_size"])
        self.sample_size = int(genpars["sample_size"])
        self.c_dim = int(genpars["c_dim"])
        self.image_shape = [self.image_size, self.image_size, self.c_dim]

        self.z_dim = int(genpars["z_dim"])

        self.gf_dim = int(genpars["gf_dim"])
        self.df_dim = int(genpars["df_dim"])

        self.gfc_dim = int(genpars["gfc_dim"])
        self.dfc_dim = int(genpars["dfc_dim"])

        self.lam = float(genpars["lam"])

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [
            batch_norm(
                name="d_bn{}".format(
                    i,
                )
            )
            for i in range(4)
        ]

        log_size = int(math.log(self.image_size) / math.log(2))
        self.g_bns = [
            batch_norm(
                name="g_bn{}".format(
                    i,
                )
            )
            for i in range(log_size)
        ]

        self.build_model()

        self.model_name = "DCGAN.model"

    def build_model(self):
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        self.images = tf.compat.v1.placeholder(
            tf.float32, [None] + self.image_shape, name="real_images"
        )

        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.z_dim], name="z")
        self.z_sum = tf.compat.v1.summary.histogram("z", self.z)

        self.G = self.generator(self.z)

        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.compat.v1.summary.histogram("d", self.D)
        self.d__sum = tf.compat.v1.summary.histogram("d_", self.D_)
        self.G_sum = tf.compat.v1.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)
            )
        )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)
            )
        )
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)
            )
        )

        self.d_loss_real_sum = tf.compat.v1.summary.scalar(
            "d_loss_real", self.d_loss_real
        )
        self.d_loss_fake_sum = tf.compat.v1.summary.scalar(
            "d_loss_fake", self.d_loss_fake
        )

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.compat.v1.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.compat.v1.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if "d_" in var.name]
        self.g_vars = [var for var in t_vars if "g_" in var.name]

        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, self.image_shape, name="mask")
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(
                    tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images)
                )
            ),
            1,
        )
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, trainpars):

        epoch = int(trainpars["epoch"])
        learning_rate = float(trainpars["learning_rate"])
        beta1 = float(trainpars["beta1"])
        train_size = float(trainpars["train_size"])
        if train_size == 1e99:
            train_size = np.inf
        batch_size = int(trainpars["batch_size"])
        dataset = directory(str(trainpars["dataset"]))
        checkpoint_dir = directory(str(trainpars["checkpoint_dir"]))
        sample_dir = str(trainpars["sample_dir"])
        log_dir = directory(str(trainpars["log_dir"]))
        training_set_ratio = float(trainpars["training_set_ratio"])
        num_chunks = int(trainpars["num_chunks"])

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.d_loss, var_list=self.d_vars
        )
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
            self.g_loss, var_list=self.g_vars
        )
        with self.sess.as_default():
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()
        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum]
        )
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum]
        )
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(checkpoint_dir):
            print(
                """
            ======
            An existing model was found in the checkpoint directory.
            If you just cloned this repository, it's a model for exoplanetary spectra 
            creating repackaging 10 millions spectra in groups of 10 thousands.
            If you want to train a new model from scratch,
            delete the checkpoint directory or specify a different
            --checkpoint_dir argument.
            ======
            """
            )
        else:
            print(
                """
            ======
            An existing model was not found in the checkpoint directory.
            Initializing a new one.
            ======
            """
            )

        for epoch in xrange(epoch):
            # print(
            #     f"Loading dataset number: {[epoch*chunk_num for chunk_num in range(num_chunks)]}"
            # )
            data = get_aspa_dataset_from_hdf5(dataset, num_chunks)
            np.random.shuffle(data)
            assert len(data) > 0
            batch_idxs = min(len(data), train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch = data[idx * batch_size : (idx + 1) * batch_size]
                # batch = [get_spectral_matrix(batch_file, size=self.image_size - 10)
                #          for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [batch_size, self.z_dim]).astype(
                    np.float32
                )

                # Update D network
                _, summary_str = self.sess.run(
                    [d_optim, self.d_sum],
                    feed_dict={
                        self.images: batch_images,
                        self.z: batch_z,
                        self.is_training: True,
                    },
                )
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum],
                    feed_dict={self.z: batch_z, self.is_training: True},
                )
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum],
                    feed_dict={self.z: batch_z, self.is_training: True},
                )
                self.writer.add_summary(summary_str, counter)
                with self.sess.as_default():
                    errD_fake = self.d_loss_fake.eval(
                        {self.z: batch_z, self.is_training: False}
                    )
                    errD_real = self.d_loss_real.eval(
                        {self.images: batch_images, self.is_training: False}
                    )
                    errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print(
                    "Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                        epoch,
                        idx,
                        batch_idxs,
                        time.time() - start_time,
                        errD_fake + errD_real,
                        errG,
                    )
                )

                #        if np.mod(counter, 1000) == 1:
                #          samples, d_loss, g_loss = self.sess.run(
                #            [self.G, self.d_loss, self.g_loss],
                #            feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                #          )
                #          save_images(samples, [8, 8],
                #                      './samples/train_{:02d}_{:04d}.pdf'.format(epoch, idx))
                #          print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                if np.mod(counter, 1000) == 2:
                    self.save(checkpoint_dir, counter)

    def complete(self, comppars, X, parfile=None, sigma=0.0):
        """
        Finds the best representation that can complete any missing
        part of the ASPA code.

        Input: any spectrum correctly converted into an ASPA code
        """
        outDir = directory(comppars["outDir"])
        maskType = comppars["maskType"]
        centerScale = comppars["centerScale"]
        nIter = int(comppars["nIter"])
        outInterval = int(comppars["outInterval"])
        approach = comppars["approach"]
        make_corner = bool(comppars["make_corner"])

        beta1 = comppars["beta1"]
        beta2 = comppars["beta2"]
        lr = comppars["lr"]
        eps = comppars["eps"]

        hmcL = int(comppars["hmcL"])
        hmcEps = comppars["hmcEps"]
        hmcBeta = comppars["hmcBeta"]
        hmcAnneal = comppars["hmcAnneal"]
        checkpoint_dir = directory(comppars["checkpointDir"])

        build_directories(comppars)

        if type(X) == dict:
            X_to_split = copy.deepcopy(X)
        elif type(X) == str:
            if X[-3:] == "dat":
                X_to_split = str(X)
        else:
            X_to_split = np.array(X)

        try:
            if type(X) != np.ndarray:
                true_spectrum = X
            else:
                true_spectrum = None
        except IOError:
            true_spectrum = None

        with self.sess.as_default():
            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

        isLoaded = self.load(checkpoint_dir)
        assert isLoaded

        grids = Grids()
        wnw_grid = grids.wnw_grid

        nImgs = self.batch_size

        batch_idxs = int(np.ceil(nImgs / self.batch_size))
        if maskType == "random":
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif maskType == "center":
            assert centerScale <= 0.5
            mask = np.ones(self.image_shape)
            l = int(self.image_size * centerScale)
            u = int(self.image_size * (1.0 - centerScale))
            mask[l:u, l:u, :] = 0.0
        elif maskType == "left":
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:, :c, :] = 0.0
        elif maskType == "full":
            mask = np.ones(self.image_shape)
        elif maskType == "grid":
            mask = np.zeros(self.image_shape)
            mask[::4, ::4, :] = 1.0
        elif maskType == "lowres":
            mask = np.zeros(self.image_shape)
        elif maskType == "parameters":
            assert centerScale <= 0.5
            mask = np.ones(self.image_shape)
            mask[-3:, :, :] = 0.0
            mask[:, -3:, :] = 0.0
            mask[-10:, -10:, :] = 0.0
        elif maskType == "wfc3":
            assert centerScale <= 0.5
            m_size = self.image_size - 10
            mask = np.ones(self.image_shape)
            fake_spec = np.ones(m_size**2)
            fake_spec[:334] = 0.0
            fake_spec[384:] = 0.0
            fake_spec = fake_spec.reshape((m_size, m_size))
            mask[:m_size, :m_size, 0] = fake_spec
            mask[-8:, :, :] = 0.0
            mask[:, -10:, :] = 0.0
            mask[-10:, -10:, :] = 0.0
        else:
            assert False

        for idx in xrange(0, batch_idxs):
            l = idx * self.batch_size
            u = min((idx + 1) * self.batch_size, nImgs)
            batchSz = u - l
            if type(X) != str:
                Xtrue = get_spectral_matrix(X, size=self.image_size - 10)
                Xt = get_test_image(
                    X, sigma=sigma, size=self.image_size, batch_size=self.batch_size
                )
            else:
                Xtrue = get_spectral_matrix(
                    X, parfile=parfile, size=self.image_size - 10
                )
                Xt = get_test_image(
                    X,
                    sigma=sigma,
                    size=self.image_size,
                    batch_size=self.batch_size,
                    parfile=parfile,
                )
            spec_parameters = get_parameters(Xtrue, size=self.image_size)

            batch = Xt
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size - batchSz)), (0, 0), (0, 0), (0, 0))
                batch_images = np.pad(batch_images, padSz, "constant")

                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            nImgs = 1
            nRows = int(np.sqrt(nImgs))
            nCols = int(np.sqrt(nImgs))
            #      save_images(batch_images[:nImgs, :, :, :], [nRows, nCols],
            #                  os.path.join(config.outDir, 'before.pdf'))

            plt.imshow(
                Xtrue[:, :, 0],
                cmap="gist_gray",
            )
            plt.axis("off")

            plt.savefig(
                os.path.join(outDir, "before.png"),
                dpi=900,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
            resize(os.path.join(outDir, "before.png"))

            masked_images = np.multiply(batch_images, mask)
            #      save_images(masked_images[:nImgs, :, :, :], [nRows, nCols],
            #                  os.path.join(config.outDir, 'masked.pdf'))

            plt.imshow(
                masked_images[0, :, :, 0],
                cmap="gist_gray",
            )
            plt.axis("off")
            plt.savefig(
                os.path.join(outDir, "masked.png"),
                dpi=900,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()
            resize(os.path.join(outDir, "masked.png"))

            for img in range(batchSz):
                with open(
                    os.path.join(outDir, "logs/hats_{:02d}.log".format(img)), "a"
                ) as f:
                    f.write(
                        "iter loss "
                        + " ".join(["z{}".format(zi) for zi in range(self.z_dim)])
                        + "\n"
                    )

            for i in xrange(nIter):

                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.images: batch_images,
                    self.is_training: False,
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(
                        os.path.join(outDir, "logs/hats_{:02d}.log".format(img)), "ab"
                    ) as f:
                        f.write("{} {} ".format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img : img + 1])

                if i % outInterval == 0:
                    prediction_file = open(
                        outDir + "predictions/prediction_{:04d}.txt".format(i), "w"
                    )

                    ranges = []
                    ground_truths = []
                    gan_avg = []
                    gan_p_err = []
                    gan_m_err = []

                    if type(X) == str:
                        """
                        If the input spectrum is synthetic, you know the parameters array
                        and you want to compare the real value with the retrieved one, if
                        your spectrum does not contain a molecule, the default value is fixed
                        to -7.9
                        """
                        pp = ParameterParser()
                        pp.read(parfile)
                        real_pars = pp.full_dict()

                        real_tp = float(real_pars["Atmosphere"]["tp_iso_temp"])
                        real_rp = float(real_pars["Planet"]["radius"])
                        real_mp = float(real_pars["Planet"]["mass"])
                        atm_active_gases = np.array(
                            [
                                gas.upper()
                                for gas in real_pars["Atmosphere"]["active_gases"]
                            ]
                        )
                        atm_active_gases_mixratios = np.array(
                            real_pars["Atmosphere"]["active_gases_mixratios"]
                        )
                        real_mol = check_molecule_existence(
                            ["CO", "CO2", "H2O", "CH4"],
                            atm_active_gases_mixratios,
                            atm_active_gases,
                            default=-7.9,
                        )
                        ground_truths = np.array(real_mol + [real_rp, real_mp, real_tp])

                    elif true_spectrum != None and type(X) != str:
                        h2o = np.log10(true_spectrum["param"]["h2o_mixratio"])
                        ch4 = np.log10(true_spectrum["param"]["ch4_mixratio"])
                        co2 = np.log10(true_spectrum["param"]["co2_mixratio"])
                        co = np.log10(true_spectrum["param"]["co_mixratio"])
                        rp = true_spectrum["param"]["planet_radius"] / RJUP
                        mp = true_spectrum["param"]["planet_mass"] / MJUP
                        tp = true_spectrum["param"]["temperature_profile"]
                        ground_truths = np.array([co, co2, h2o, ch4, rp, mp, tp])
                        real_mol = np.zeros(4)
                    else:
                        ground_truths = np.array([None] * 7)
                        real_mol = np.zeros(4)

                    parameters = ["CO", "CO2", "H2O", "CH4", "Rp", "Mp", "Tp"]
                    labels = [
                        "$\log{CO}$",
                        "$\log{CO_2}$",
                        "$\log{H_2O}$",
                        "$\log{CH_4}$",
                        "$R_p (R_j)$",
                        "$M_p (M_j)$",
                        "$T_p$",
                    ]

                    all_hists = []
                    for mol in parameters:
                        (
                            prediction_file,
                            gan_avg,
                            gan_p_err,
                            gan_m_err,
                            ranges,
                            all_hists,
                        ) = histogram_par(
                            mol,
                            G_imgs,
                            batchSz,
                            self.image_size,
                            ground_truths,
                            all_hists,
                            prediction_file,
                            gan_avg,
                            gan_p_err,
                            gan_m_err,
                            ranges,
                        )

                    all_hists = np.array(all_hists).T

                    if make_corner:
                        make_corner_plot(
                            all_hists, ranges, labels, ground_truths, comppars, i
                        )

                    """
                    Plot histograms
                    """
                    hist_dict = {}
                    f, ax = plt.subplots(2, 4, figsize=(21, 15))
                    all_hists = all_hists.T
                    ii = 0
                    jj = 0
                    for his in range(len(all_hists)):
                        if his == 4:
                            ii = 1
                            jj = 4
                        hist_dict[labels[his]] = {}
                        weights = np.ones_like(all_hists[his]) / float(
                            len(all_hists[his])
                        )
                        hist_dict[labels[his]]["histogram"] = all_hists[his]
                        hist_dict[labels[his]]["weights"] = weights
                        hist_dict[labels[his]]["bins"] = ranges[his]
                        ax[ii, his - jj].hist(
                            all_hists[his],
                            bins=np.linspace(min(ranges[his]), max(ranges[his]), 20),
                            color="firebrick",
                            weights=weights,
                        )
                        #            ax[his].set_ylim(0, 1)
                        ax[ii, his - jj].set_xlim(min(ranges[his]), max(ranges[his]))
                        ax[ii, his - jj].axvline(
                            gan_avg[his], c="g", label="ExoGAN mean"
                        )
                        ax[ii, his - jj].axvline(
                            ground_truths[his], c="b", label="Input value"
                        )
                        ax[ii, his - jj].set_xlabel(
                            labels[his]
                            + " = $%1.2f_{-%1.2f}^{%1.2f}$"
                            % (gan_avg[his], gan_m_err[his], gan_p_err[his])
                        )
                        if his == 3:
                            ax[ii, his - jj].legend()
                        #            ax[his].annotate('$%1.2f_{-%1.2f}^{%1.2f}$' % (gan_avg[his], gan_p_err[his], gan_m_err[his]),
                        #               bbox=dict(boxstyle="round4", fc="w", alpha=0.5),
                        #               xy=(gan_avg[his], max(weights)*(0.9)),
                        #               xycoords='data')
                        ax[ii, his - jj].axvline(
                            gan_avg[his] + gan_p_err[his], c="k", linestyle="--"
                        )
                        ax[ii, his - jj].axvline(
                            gan_avg[his] - gan_m_err[his], c="k", linestyle="--"
                        )
                    ax[-1, -1].axis("off")
                    plt.subplots_adjust(right=1.2)

                    histName = os.path.join(
                        outDir, "histograms/all_par/{:04d}.pdf".format(i)
                    )
                    plt.savefig(histName, bbox_inches="tight")
                    plt.close()
                    histpickle = os.path.join(
                        outDir, "histograms/all_par/histogram.pickle"
                    )
                    with open(histpickle, "wb") as fp:
                        pickle.dump(hist_dict, fp)

                    real_spec = Xtrue[: self.image_size, : self.image_size, :]
                    real_spec = real_spec[:23, :23, 0].flatten()

                    chi_square = []
                    spectra = []
                    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
                    for k in range(batchSz):
                        spectrum = G_imgs[k, : self.image_size, : self.image_size, :]
                        spectrum = spectrum[:23, :23, 0].flatten()
                        spectra.append(spectrum)
                        chi_square.append(
                            (
                                ((spectrum[:440] - real_spec[:440]) ** 2)
                                / real_spec[:440]
                            )
                            .sum()
                            .sum()
                            # chisquare(spectrum[:440], f_exp=real_spec[:440])[0]
                        )
                    best_ind = chi_square.index(min(chi_square))

                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(outDir, "hats_imgs/{:04d}.png".format(i))

                    #          save_images(G_imgs[:nImgs, :, :, :], [nRows, nCols], imgName)
                    plt.imshow(
                        G_imgs[best_ind, :, :, 0],
                        cmap="gist_gray",
                    )
                    plt.axis("off")
                    plt.savefig(imgName, dpi=900, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    resize(imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(outDir, "completed/{:04d}.png".format(i))
                    #          save_images(completed[:nImgs, :, :, :], [nRows, nCols], imgName)
                    plt.imshow(
                        completed[best_ind, :, :, 0],
                        cmap="gist_gray",
                    )
                    plt.axis("off")
                    plt.savefig(imgName, dpi=900, bbox_inches="tight", pad_inches=0)
                    plt.close()
                    resize(imgName)

                    if spectra_int_norm:
                        # Compared real spectrum with the generated one
                        spectra_int_norm(
                            Xtrue,
                            self.image_size,
                            wnw_grid,
                            batchSz,
                            G_imgs,
                            comppars,
                            i,
                        )

                    if spectra_norm:
                        # Compare spectra with original normalisation between 0 and 1
                        spectra_norm(
                            Xtrue,
                            self.image_size,
                            wnw_grid,
                            batchSz,
                            G_imgs,
                            comppars,
                            i,
                        )

                    if spectra_real_norm:
                        # Compare spectra with the normalisation factor from the real spectrum
                        spectra_real_norm(
                            Xtrue,
                            self.image_size,
                            wnw_grid,
                            batchSz,
                            G_imgs,
                            comppars,
                            i,
                        )

                if approach == "adam":
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = beta1 * m_prev + (1 - beta1) * g[0]
                    v = beta2 * v_prev + (1 - beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - beta1 ** (i + 1))
                    v_hat = v / (1 - beta2 ** (i + 1))
                    zhats += -np.true_divide(lr * m_hat, (np.sqrt(v_hat) + eps))
                    zhats = np.clip(zhats, -1, 1)

                elif approach == "hmc":
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
                        logprob_old = (
                            hmcBeta * loss_old[img] + np.sum(v_old[img] ** 2) / 2
                        )
                        logprob = hmcBeta * loss[img] + np.sum(v[img] ** 2) / 2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    hmcBeta *= hmcAnneal

                else:
                    assert False

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = lrelu(conv2d(image, self.df_dim, name="d_h0_conv"))
            h1 = lrelu(
                self.d_bns[0](
                    conv2d(h0, self.df_dim * 2, name="d_h1_conv"), self.is_training
                )
            )
            h2 = lrelu(
                self.d_bns[1](
                    conv2d(h1, self.df_dim * 4, name="d_h2_conv"), self.is_training
                )
            )
            h3 = lrelu(
                self.d_bns[2](
                    conv2d(h2, self.df_dim * 8, name="d_h3_conv"), self.is_training
                )
            )
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, "d_h4_lin")

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.image_size, self.image_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, "g_h0_lin", with_w=True
            )

            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            hs.append(None)
            hs[1], _, _ = conv2d_transpose(
                hs[0],
                [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                name="g_h1",
                with_w=True,
            )
            hs[1] = tf.nn.relu(self.g_bns[1](hs[1], self.is_training))

            hs.append(None)
            hs[2], _, _ = conv2d_transpose(
                hs[1],
                [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                name="g_h2",
                with_w=True,
            )
            hs[2] = tf.nn.relu(self.g_bns[2](hs[2], self.is_training))

            hs.append(None)
            hs[3], _, _ = conv2d_transpose(
                hs[2],
                [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                name="g_h3",
                with_w=True,
            )
            hs[3] = tf.nn.relu(self.g_bns[3](hs[3], self.is_training))

            hs.append(None)
            hs[4], _, _ = conv2d_transpose(
                hs[3], [self.batch_size, s_h, s_w, self.c_dim], name="g_h4", with_w=True
            )

            # for normalisations between 0 and 1 use 'tf.nn.sigmoid(hs[4])'

            return tf.nn.sigmoid(hs[4])

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(
            self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step
        )

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
