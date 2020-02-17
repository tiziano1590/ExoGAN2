# For the second part of the utils.py file
# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import logging
import glob
import pandas as pd
import gzip
import tensorflow as tf
import multiprocessing as mp
import copy
from PIL import Image
import imageio
import corner
from scipy.stats import chisquare
from scipy.interpolate import interp1d

from tensorflow.python.framework import ops
from exogan.tools import ReportInterface
from exogan.libraries import Grids
from exogan.parameter import ParameterParser

RJUP = 6.9911e7
MJUP = 1.898e27


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def directory(name):
    if name[-1] == '/':
        return name
    else:
        return name + '/'


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def planckel(wavelength, temperature=5800):
    """ Planck function in wavelength for radiant emittance.

    Args:
        | wavelength (np.array[N,]):  wavelength vector in  [um]
        | temperature (float):  Temperature scalar in [K]

    Returns:
        | (np.array[N,]):  spectral radiant emitance in W/(m^2.um)

    Raises:
        | No exception is raised.
    """
    return 3.7418301e8 / (wavelength ** 5 * (np.exp(14387.86e0 / (wavelength * temperature)) - 1.))


def quantile_corner(x, q, weights=None):
    """

  * Taken from corner.py
  __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
  __copyright__ = "Copyright 2013-2015 Daniel Foreman-Mackey"

  Like numpy.percentile, but:

  * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
  * scalar q not supported (q must be iterable)
  * optional weights on x

  """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


"""
SECOND PART
"""

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path):
    return imageio.imread(path, mode='RGB').astype(np.float)


def merge_images(images):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    img = merge(images, size)
    return imageio.imsave(path, (255 * img).astype(np.uint8))


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w],
                               [resize_w, resize_w])


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


"""
Deep Learning operations
"""


class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.compat.v1.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)


def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.
    For brevity, let `x = `, `z = targets`.  The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
        :param targets: targets
        :param preds: predictions
        :param name: label associated to binary_cross_entropy function
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                                (1. - targets) * tf.log(1. - preds + eps)))


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv


def conv2d_transpose(input_, output_shape,
                     k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                           tf.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output_size],
                                         initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


"""
Utilities from ExoGAN v1.0
"""


def ASPA_read(G, layer, mol, imgsz):
    """
    Interprets G ASPA's output

    Inputs:
      G:      trained neural network
      layer:  single prediction
      mol:    molecule option
      imgsz:  size of the input matrix

    return:
      molecule value prediction
    """
    aspas = {
        'CO': G[layer, :imgsz - 10, -2, 0],
        'CO2': G[layer, :imgsz - 10, -3, 0],
        'H2O': G[layer, -3:, -3:, 0],
        'CH4': G[layer, :imgsz - 10, -1, 0],
        'Rp': G[layer, -2, :imgsz - 10, 0],
        'Mp': G[layer, -3, :imgsz - 10, 0],
        'Tp': G[layer, -1, :imgsz - 10, 0]
    }
    return aspas[mol]


def build_directories(config):
    make_dir('hats_imgs', config)
    make_dir('histograms/all_par', config)
    make_dir('completed', config)
    make_dir('logs', config)
    make_dir('predictions', config)


def check_molecule_existence(mol_list, array, array_names, default=-7.9):
    real_mol = []
    for mol in mol_list:
        if mol in array_names:
            index = np.where(array_names == mol)[0]
            real_mol.append(np.log10(array[index]))
        else:
            real_mol.append(default)
    return real_mol


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)


def get_parameters(X, size=33):
    parameters = []
    # Add CO2
    parameters.append(np.mean(X[:size - 3, -3, 0]))
    # Add CO
    parameters.append(np.mean(X[:size - 3, -2, 0]))
    # Add CH4
    parameters.append(np.mean(X[:size - 3, -1, 0]))
    # Add H2O
    parameters.append(np.mean(X[-10:, -10:, 0]))
    # Add Mp
    parameters.append(np.mean(X[-3, :size - 3, 0]))
    # Add Rp
    parameters.append(np.mean(X[-2, :size - 3, 0]))
    # Add Tp
    parameters.append(np.mean(X[-1, :size - 3, 0]))

    parameters = np.array(parameters)
    return parameters


def get_spectra(limit=None):
    # if not os.path.exists('../large_files'):
    #   os.mkdir('../large_files')

    print("Reading in and transforming data...")
    df = pd.read_csv('spectra_train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)
    X = data[:, 1:]
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_spectral_library(split=0.1):
    print("Reading in and transforming data...")
    #  data_set = glob('/mnt/test_set/*pkgz')
    data_set = glob('../../TauREx_deep/Output/test_set/*pkgz')
    #  data_set = glob('./test_spectra/*pkgz')
    np.random.shuffle(data_set)
    np.random.shuffle(data_set)
    data_train = data_set[:int(len(data_set) * (1 - split))]
    data_test = data_set[int(len(data_set) * (1 - split)):]
    np.savetxt('data_train.dat', data_train, fmt="%s")
    np.savetxt('data_test.dat', data_test, fmt="%s")
    return data_train


def get_spectral_matrix(path, parfile=None, size=23):
    grids = Grids()
    wnw_grid = grids.wnw_grid
    # global normalisation over the entire dataset
    norm_bounds = np.array([0.63, 0.77, 0.88, 1.0, 1.1, 1.25,
                            1.8, 2.13, 3.96, 4.56, 6.33, 7.21, 10.0])
    norm_idx = [0, 162, 195, 208, 241, 255, 318, 334, 371,
                384, 394, 406, 420, 440, -1]

    global_maximum = 0.0301803471629
    global_minimum = 0.00650370296635
    half_row = 14
    if type(path) == str and path[-3:] == 'dat':

        # TODO -- correction for stellar radius
        pp = ParameterParser()
        pp.read(parfile)
        real_pars = pp.full_dict()

        star_radius = float(real_pars['Star']['radius'])
        radius_fac = star_radius ** 2
        spec = np.genfromtxt(path)

        _, wnw_min = find_nearest(wnw_grid, max(spec[:, 0]))
        _, wnw_max = find_nearest(wnw_grid, min(spec[:, 0]))

        spectrum, spec_err = np.zeros(len(wnw_grid)), np.zeros(len(wnw_grid))
        spectrum[wnw_min:wnw_max + 1] = spec[:, 1] * radius_fac
        spec_err[wnw_min:wnw_max + 1] = spec[:, 2] * radius_fac

        spectrum = np.random.normal(spectrum, spec_err)

        max_s = np.mean(spectrum) / global_maximum
        mean_param = [max_s] * half_row
        mean_param = np.array(mean_param)
        param_list = []
        planet_temperature = float(real_pars['Atmosphere']['tp_iso_temp']) / 2e3
        planet_radius = float(real_pars['Planet']['radius']) / 1.5
        planet_mass = float(real_pars['Planet']['mass']) / 2.0
        atm_active_gases = np.array([gas.upper() for gas in real_pars['Atmosphere']['active_gases']])
        atm_active_gases_mixratios = np.array(real_pars['Atmosphere']['active_gases_mixratios'])

        atm_active_gases_mixratios = -np.log10(atm_active_gases_mixratios) / 8.
        if 'H2O' in atm_active_gases:
            index = np.where(atm_active_gases == 'H2O')[0]
            h2o_mixratio = atm_active_gases_mixratios[index]
        else:
            h2o_mixratio = 1e-8
        if 'CH4' in atm_active_gases:
            index = np.where(atm_active_gases == 'CH4')[0]
            ch4_mixratio = atm_active_gases_mixratios[index]
        else:
            ch4_mixratio = 1e-8
        if 'CO2' in atm_active_gases:
            index = np.where(atm_active_gases == 'CO2')[0]
            co2_mixratio = atm_active_gases_mixratios[index]
        else:
            co2_mixratio = 1e-8
        if 'CO' in atm_active_gases:
            index = np.where(atm_active_gases == 'CO')[0]
            co_mixratio = atm_active_gases_mixratios[index]
        else:
            co_mixratio = 1e-8

    elif type(path) == np.ndarray:
        spectrum = path
        param_list = [0] * 7

        param_list = np.array(param_list)

        half_row = 14
        max_s = np.mean(spectrum) / global_maximum
        mean_param = [max_s] * half_row
        mean_param = np.array(mean_param)
    elif type(path) == dict:
        spec = path
        param_list = []
        for key in spec['param'].keys():
            value = spec['param'][key]

            if key == 'temperature_profile':
                value /= 2e3
            elif key == 'planet_radius':
                value /= (1.5 * RJUP)
            elif key == 'planet_mass':
                value /= (2.0 * MJUP)
            elif key == 'h2o_mixratio':
                value = -np.log10(value) / 8.
            elif key == 'ch4_mixratio':
                value = -np.log10(value) / 8.
            elif key == 'co_mixratio':
                value = -np.log10(value) / 8.
            elif key == 'co2_mixratio':
                value = -np.log10(value) / 8.

            param_list.append(value)

        param_list = np.array(param_list)

        spectrum = spec['data']['spectrum']
        half_row = 14
        max_s = np.mean(spectrum) / global_maximum
        mean_param = [max_s] * half_row
        mean_param = np.array(mean_param)

    new_size = int(size + 3 + (len(norm_idx) - 1) / 2)
    new_row = np.zeros((new_size, new_size, 1))

    norm_spectrum = np.array(spectrum)

    for i in range(len(norm_idx) - 1):
        frag = spectrum[norm_idx[i]:norm_idx[i + 1]]
        if i == len(norm_idx) - 2:
            frag = frag[:-4]

        minf = np.min(frag)
        maxf = np.max(frag)

        if (minf == maxf):
            norm_spectrum[norm_idx[i]:norm_idx[i + 1]] = 0.
        elif i == len(norm_idx) - 2:
            norm_spectrum[norm_idx[i]:norm_idx[i + 1]][:-4] = (frag - minf) / (maxf - minf)
            norm_spectrum[norm_idx[i]:norm_idx[i + 1]][-4:] = 0
        else:
            norm_spectrum[norm_idx[i]:norm_idx[i + 1]] = (frag - minf) / (maxf - minf)
        if i < (len(norm_idx) - 1) / 2:
            new_row[0:12, size + i, 0] = maxf / global_maximum
            if minf > 0:
                new_row[12:size, size + i, 0] = global_minimum / minf
            else:
                new_row[12:size, size + i, 0] = 0.
        else:
            new_row[size - 7 + i, 0:12, 0] = maxf / global_maximum
            if minf > 0:
                new_row[size - 7 + i, 12:size, 0] = global_minimum / minf
            else:
                new_row[size - 7 + i, 12:size, 0] = 0

    row = np.concatenate((norm_spectrum, mean_param))
    row = row.reshape(size, size)

    new_row[:size, :size, 0] = row

    try:
        # Add CO2
        new_row[:size, -3, 0] = param_list[5]
        # Add CO
        new_row[:size, -2, 0] = param_list[6]
        # Add CH4
        new_row[:size, -1, 0] = param_list[2]
        # Add H2O
        new_row[size:, size:, 0] = param_list[4]
        # Add Mp
        new_row[-3, :size, 0] = param_list[0]
        # Add Rp
        new_row[-2, :size, 0] = param_list[3]
        # Add Tp
        new_row[-1, :size, 0] = param_list[1]
    except:
        if path[-3:] == 'dat':
            # Add CO2
            new_row[:size, -3, 0] = co2_mixratio
            # Add CO
            new_row[:size, -2, 0] = co_mixratio
            # Add CH4
            new_row[:size, -1, 0] = ch4_mixratio
            # Add H2O
            new_row[size:, size:, 0] = h2o_mixratio
            # Add Mp
            new_row[-3, :size, 0] = planet_mass
            # Add Rp
            new_row[-2, :size, 0] = planet_radius
            # Add Tp
            new_row[-1, :size, 0] = planet_temperature

    return new_row


def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def get_aspa_dataset_from_hdf5(train_path, training_set_ratio):
    train_list = np.array(glob.glob(train_path + "*.h5"))
    np.random.shuffle(train_list)
    full_length = int(len(train_list)*training_set_ratio)
    train_list = train_list[:full_length]

    # print("Loading  h5 file to python dictionary...")
    final_dict = {}

    def hdf5_loader(ii):
        print("Loading dataset number %d" % (ii + 1))
        data = load_dict_from_hdf5(train_list[ii])
        return data

    list_of_dicts = parmap(hdf5_loader, range(len(train_list)), mp.cpu_count())

    for subdict in list_of_dicts:
        final_dict.update(subdict)

    keys = list(final_dict.keys())
    # print(keys)

    print("Turning spectral array into a ASPA matrix...")
    ASPAs = []
    for jj in range(len(keys)):
        aspa = get_spectral_matrix(final_dict[keys[jj]])
        ASPAs.append(aspa)
    ASPAs = np.array(ASPAs)
    return ASPAs


def get_test_image(X, sigma=0.0, size=33, batch_size=64, parfile=None, wfc3=False):
    batch = []
    if type(X) == dict:
        X_to_split = copy.deepcopy(X)
        X_to_split = X_to_split['data']['spectrum']
        test = lambda x: get_spectral_matrix(x, parfile=None)
    elif type(X) == str:

        if X[-3:] == 'dat':
            X_to_split = np.genfromtxt(X)[:, 1]
            test = lambda x: get_spectral_matrix(x, parfile=parfile)
    else:
        X_to_split = np.array(X)
        test = lambda x: get_spectral_matrix(x)

    new_zeros = np.zeros((size, size, 1))
    for i in range(batch_size):
        output = test(X)
        new_zeros[:size, :size, 0] = output[:size, :size, 0]
        batch.append(new_zeros)

    batch = np.array(batch)
    return batch


def getpar(parser, sec, par, type=None):
    # get parameter from user defined parser. If parameter is not found there, load the default parameter
    # the default parameter file parser is self.default_parser, defined in init
    try:
        if type == None:
            return parser.get(sec, par)
        elif type == 'float':
            return parser.getfloat(sec, par)
        elif type == 'bool':
            return parser.getboolean(sec, par)
        elif type == 'int':
            return parser.getint(sec, par)
        elif type == 'list-str':
            l = parser.get(sec, par).split(',')
            return [str(m).strip() for m in l]
        elif type == 'list-float':
            l = parser.get(sec, par).split(',')
            return [float(m) for m in l]
        elif type == 'list-int':
            l = parser.get(sec, par).split(',')
            return [int(m) for m in l]
        else:
            logging.error(
                'Cannot set parameter %s in section %s. Parameter type %s not recognized. Set to None'(par, sec, type))
            return None
    except:
        logging.error('Cannot set parameter %s in section %s. Set to None' % (par, sec))
        return None


def histogram_par(mol, G, batchsz, imgsz, ground_truths, all_hists, prediction_file, gan_avg, gan_p_err, gan_m_err,
                  ranges):
    def mult(mol):
        functs = {
            'CO': [-8., 0, (-8, 0)],
            'CO2': [-8., 1, (-8, 0)],
            'H2O': [-8., 2, (-8, 0)],
            'CH4': [-8., 3, (-8, 0)],
            'Rp': [1.5, 4, (0.8, 1.5)],
            'Mp': [2., 5, (0.8, 2.1)],
            'Tp': [2e3, 6, (1000., 2100.)]
        }
        return functs[mol]

    histospec = []
    for layer in range(batchsz):
        inter = np.mean(ASPA_read(G, layer, mol, imgsz))
        if not np.isnan(inter):
            histospec.append(inter)
    histospec = np.array(histospec)

    histospec = (histospec) * mult(mol)[0]
    all_hists.append(histospec)

    q_16, q_50, q_84 = quantile_corner(histospec, [0.16, 0.5, 0.84])
    gan_mean = q_50
    gan_sigma_m = q_50 - q_16
    gan_sigma_p = q_84 - q_50
    gan_avg.append(gan_mean)
    gan_p_err.append(gan_sigma_p)
    gan_m_err.append(gan_sigma_m)
    #  print(mol, gan_mean, gan_sigma_m, gan_sigma_p, ground_truths[mult(mol)[1]])
    #  sys.exit()
    prediction_file.write('%3s\t%1.7g\t%1.7g\t%1.7g\t%1.7g\n' %
                          (mol, gan_mean, gan_sigma_m, gan_sigma_p, ground_truths[mult(mol)[1]]))

    ranges.append(mult(mol)[2])
    return prediction_file, gan_avg, gan_p_err, gan_m_err, ranges, all_hists


def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)


def imsave(images, size, path):
    img = merge(images, size)
    #  return plt.imsave(path, img, cmap='spectral')
    return scipy.misc.imsave(path, (255 * img).astype(np.uint8))


def inverse_transform(images):
    return (images)


def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while 1:
        data = file.read()
        print(data)
        if data == "":
            break
        buffer += data

    object = pickle.loads(buffer)
    file.close()
    return object


def make_corner_plot(all_hists, ranges, labels, ground_truths, comppars, index):
    make_dir('histograms/corner', comppars)

    all_corner = corner(all_hists,
                        range=ranges,
                        smooth=0.5,
                        labels=labels,
                        quantiles=[0.16, 0.5, 0.84],
                        truths=ground_truths,
                        show_titles=True,
                        plot_contours=False,
                        fill_contours=False)
    #
    histName = os.path.join(comppars['outDir'], 'histograms/corner/{:04d}.pdf'.format(index))
    plt.savefig(histName)

    plt.close()


def make_dir(name, config):
    # Works on python 2.7, where exist_ok arg to makedirs isn't available.
    p = os.path.join(config['outDir'], name)
    if not os.path.exists(p):
        os.makedirs(p)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def quantile_corner(x, q, weights=None):
    """

    * Taken from corner.py
    __author__ = "Dan Foreman-Mackey (danfm@nyu.edu)"
    __copyright__ = "Copyright 2013-2015 Daniel Foreman-Mackey"

    Like numpy.percentile, but:

    * Values of q are quantiles [0., 1.] rather than percentiles [0., 100.]
    * scalar q not supported (q must be iterable)
    * optional weights on x

    """
    if weights is None:
        return np.percentile(x, [100. * qi for qi in q])
    else:
        idx = np.argsort(x)
        xsorted = x[idx]
        cdf = np.add.accumulate(weights[idx])
        cdf /= cdf[-1]
        return np.interp(q, cdf, xsorted).tolist()


def real_wfc3(path, star=0.756):
    planet = np.genfromtxt(path)

    wnw_grid = planet[:, 0]
    spectrum = planet[:, 1] * star ** 2

    interp = interp1d(wnw_grid, spectrum, fill_value="extrapolate")

    nn_grid = np.genfromtxt('./wnw_grid.txt')

    _, lmax = find_nearest(nn_grid, wnw_grid[0])
    _, lmin = find_nearest(nn_grid, wnw_grid[-1])

    find_nearest(nn_grid, 1.8)

    nn_lam = nn_grid[334:384][::-1]

    wfc3_spectrum = interp(nn_lam)

    nn_spectrum = np.zeros(len(nn_grid))

    nn_spectrum[334:384] = wfc3_spectrum[::-1]
    return nn_spectrum


def recon_spectrum(spec, size=23):
    norm_bounds = np.array([0.63, 0.77, 0.88, 1.0, 1.1, 1.25, 1.8, 2.13, 3.96, 4.56, 6.33, 7.21, 10.0])
    norm_idx = [0, 162, 195, 208, 241, 255, 318, 334, 371, 384, 394, 406, 420, 440, -1]
    global_maximum = 0.0301803471629
    global_minimum = 0.00650370296635

    in_spec = spec[:size, :size, 0]
    in_spec = in_spec.flatten()

    for i in range(len(norm_idx) - 1):
        frag = in_spec[norm_idx[i]:norm_idx[i + 1]]
        if i == len(norm_idx) - 2:
            frag = frag[:-4]
        if i < 7:
            maxf = np.mean(spec[0:12, size + i, 0]) * global_maximum
            minf = global_minimum / np.mean(spec[12:size, size + i, 0])
        else:
            maxf = np.mean(spec[size - 7 + i, 0:12, 0]) * global_maximum
            minf = global_minimum / np.mean(spec[size - 7 + i, 12:size, 0])

        const_control = (norm_idx[i] + norm_idx[i + 1]) / 2

        if (in_spec[norm_idx[i]] == in_spec[int(const_control)]):
            in_spec[norm_idx[i]:norm_idx[i + 1]] = 0.
        elif i == len(norm_idx) - 2:
            in_spec[norm_idx[i]:norm_idx[i + 1]][:-4] = frag * (maxf - minf) + minf
            in_spec[norm_idx[i]:norm_idx[i + 1]][-4:] = 0
        else:
            in_spec[norm_idx[i]:norm_idx[i + 1]] = frag * (maxf - minf) + minf
    return in_spec


def resize(img_name):
    img = Image.open(img_name)
    new_img = img.resize((256, 256))
    new_img.save(img_name)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def spectra_int_norm(Xtrue, imgsz, wnw_grid, batchSz, G_imgs, comppars, i):
    """
    IN-BUILT
    """
    make_dir('spectra/inbuilt/all', comppars)
    make_dir('spectra/inbuilt/best', comppars)

    real_spec = Xtrue[:imgsz, :imgsz, :]

    real_spec = recon_spectrum(real_spec)

    chi_square = []
    spectra = []
    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
    for k in range(batchSz):
        spectrum = G_imgs[k, :imgsz, :imgsz, :]
        spectrum = recon_spectrum(spectrum)
        spectra.append(spectrum)
        chi_square.append(chisquare(spectrum[:len(wnw_grid) - 4], f_exp=real_spec[:len(wnw_grid) - 4])[0])
        ax.plot(wnw_grid[:-4], spectrum[:len(wnw_grid) - 4])
    ax.plot(wnw_grid[:-4], real_spec[:len(wnw_grid) - 4], '-k', label='real spectrum')
    #          red_patch = mpatches.Patch(color='red', label='real spectrum')
    #          col_patch = mpatches.Patch(color='rainbow', label='generated spectra')
    ax.legend()

    ax.set_ylabel(r'$R_p^2/R_s^2$')
    ax.set_xlabel('Wavelength $(\mu m)$')

    histName = os.path.join(comppars['outDir'],
                            'spectra/inbuilt/all/{:04d}.pdf'.format(i))
    plt.xscale('log')
    plt.xticks([1, 10], ['1', '10'])
    #          ax.set_ylim(global_minimum, global_maximum)
    plt.savefig(histName, bbox_inches='tight')
    plt.close()

    best_ind = chi_square.index(min(chi_square))
    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
    ax.plot(wnw_grid[:-4], spectra[best_ind][:len(wnw_grid) - 4], 'r-', label='generated spectrum')
    ax.plot(wnw_grid[:-4], real_spec[:len(wnw_grid) - 4], '-k', label='real spectrum')
    #          red_patch = mpatches.Patch(color='red', label='real spectrum')
    #          col_patch = mpatches.Patch(color='rainbow', label='generated spectra')
    ax.legend()

    ax.set_ylabel(r'$R_p^2/R_s^2$')
    ax.set_xlabel('Wavelength $(\mu m)$')

    histName = os.path.join(comppars['outDir'],
                            'spectra/inbuilt/best/{:04d}.pdf'.format(i))
    plt.xscale('log')
    plt.xticks([1, 10], ['1', '10'])
    #          ax.set_ylim(global_minimum, global_maximum)
    plt.savefig(histName, bbox_inches='tight')
    plt.close()


def spectra_norm(Xtrue, imgsz, wnw_grid, batchSz, G_imgs, comppars, i):
    """
    NORMALISED SPECTRA
    """
    make_dir('spectra/normalised/all', comppars)
    make_dir('spectra/normalised/best', comppars)

    real_spec = Xtrue[:imgsz, :imgsz, :]
    real_spec = real_spec[:23, :23, 0].flatten()

    chi_square = []
    spectra = []
    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
    for k in range(batchSz):
        spectrum = G_imgs[k, :imgsz, :imgsz, :]
        spectrum = spectrum[:23, :23, 0].flatten()
        spectra.append(spectrum)
        chi_square.append(chisquare(spectrum[:440], f_exp=real_spec[:440])[0])
        ax.plot(wnw_grid[:-4], spectrum[:len(wnw_grid) - 4])
    ax.plot(wnw_grid[:-4], real_spec[:len(wnw_grid) - 4], '-k', label='real spectrum')
    #          red_patch = mpatches.Patch(color='red', label='real spectrum')
    #          col_patch = mpatches.Patch(color='rainbow', label='generated spectra')
    ax.legend()

    ax.set_ylabel(r'$R_p^2/R_s^2$')
    ax.set_xlabel('Wavelength $(\mu m)$')

    histName = os.path.join(comppars['outDir'],
                            'spectra/normalised/all/{:04d}.pdf'.format(i))
    plt.xscale('log')
    plt.xticks([1, 10], ['1', '10'])
    #          ax.set_ylim(global_minimum, global_maximum)
    plt.savefig(histName, bbox_inches='tight')
    plt.close()

    best_ind = chi_square.index(min(chi_square))
    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
    ax.plot(wnw_grid[:-4], spectra[best_ind][:len(wnw_grid) - 4], 'r-', label='generated spectrum')
    ax.plot(wnw_grid[:-4], real_spec[:len(wnw_grid) - 4], '-k', label='real spectrum')
    #          red_patch = mpatches.Patch(color='red', label='real spectrum')
    #          col_patch = mpatches.Patch(color='rainbow', label='generated spectra')
    ax.legend()

    ax.set_ylabel(r'$R_p^2/R_s^2$')
    ax.set_xlabel('Wavelength $(\mu m)$')

    histName = os.path.join(comppars['outDir'],
                            'spectra/normalised/best/{:04d}.pdf'.format(i))
    plt.xscale('log')
    plt.xticks([1, 10], ['1', '10'])
    #          ax.set_ylim(global_minimum, global_maximum)
    plt.savefig(histName, bbox_inches='tight')
    plt.close()


def spectra_real_norm(Xtrue, imgsz, wnw_grid, batchSz, G_imgs, comppars, i):
    """
    WITH REAL NORMALISATION
    """
    make_dir('spectra/with_real_norm/all', comppars)
    make_dir('spectra/with_real_norm/best', comppars)

    real_spec_ori = Xtrue[:imgsz, :imgsz, :]

    real_spec = recon_spectrum(real_spec_ori)

    chi_square = []
    spectra = []
    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
    for k in range(batchSz):
        spectrum = G_imgs[k, :imgsz, :imgsz, :]
        spectrum[23:, :, 0] = real_spec_ori[23:, :, 0]
        spectrum[:, 23:, 0] = real_spec_ori[:, 23:, 0]
        spectrum = recon_spectrum(spectrum)
        spectra.append(spectrum)
        chi_square.append(chisquare(spectrum[:-8], f_exp=real_spec[:-8])[0])
        ax.plot(wnw_grid[:-8], spectrum[:len(wnw_grid) - 8])
    ax.plot(wnw_grid[:-8], real_spec[:len(wnw_grid) - 8], '-k', label='real spectrum')
    #          red_patch = mpatches.Patch(color='red', label='real spectrum')
    #          col_patch = mpatches.Patch(color='rainbow', label='generated spectra')
    ax.legend()

    ax.set_ylabel(r'$R_p^2/R_s^2$')
    ax.set_xlabel('Wavelength $(\mu m)$')

    histName = os.path.join(comppars['outDir'],
                            'spectra/with_real_norm/all/{:04d}.pdf'.format(i))
    plt.xscale('log')
    plt.xticks([1, 10], ['1', '10'])
    plt.savefig(histName, bbox_inches='tight')
    plt.close()

    best_ind = chi_square.index(min(chi_square))
    f, ax = plt.subplots(sharey=True, figsize=(12, 6))
    ax.plot(wnw_grid[:-8], spectra[best_ind][:len(wnw_grid) - 8], 'r-', label='generated spectrum')
    ax.plot(wnw_grid[:-8], real_spec[:len(wnw_grid) - 8], '-k', label='real spectrum')
    #          red_patch = mpatches.Patch(color='red', label='real spectrum')
    #          col_patch = mpatches.Patch(color='rainbow', label='generated spectra')
    ax.legend()

    ax.set_ylabel(r'$R_p^2/R_s^2$')
    ax.set_xlabel('Wavelength $(\mu m)$')

    histName = os.path.join(comppars['outDir'],
                            'spectra/with_real_norm/best/{:04d}.pdf'.format(i))
    plt.xscale('log')
    plt.xticks([1, 10], ['1', '10'])
    #          ax.set_ylim(min(real_spec[:440]), max(real_spec))
    #          ax.set_xlim(0.35, )
    plt.savefig(histName, bbox_inches='tight')
    plt.close()


def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)


def load_dict_from_hdf5(filename):
    return ReportInterface.__load_dict_from_hdf5__(filename)
