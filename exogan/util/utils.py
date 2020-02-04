# For the second part of the utils.py file
# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/utils.py
#   + License: MIT

"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import numpy as np
import sys
import math
from tqdm import tqdm
import json
import random
import pprint
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import itertools
import numpy as np
from time import gmtime, strftime
import time
import os
import stat
import pickle
import logging
import glob

from astropy.io import fits
from scipy.signal import medfilt

from libraries.library_constants import *
from libraries.library_grid import *
from classes.planet import *


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def make_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)
        
def check_fin_dir(name):
    if name[-1] == '/':
        return name
    else:
        return name+'/'


def nan_rem(arr):
    """
    DEPRECATED: Antonino's IDL function to take care of NaNs.
    Kept just as a possible example

    :param arr:
    :return: array without nans
    """
    ll = np.where(arr > 0)[0]
    nn1 = len(arr)
    indx = np.arange(nn1, dtype=int)
    indx = np.delete(indx, ll)
    print(indx)

    nn = len(indx)
    for k in range(2, nn - 3 + 1):
        dum = arr[indx[k] - 2: indx[k] + 2]
        ll = np.where((dum > 0))[0]
        nnp = len(ll)
        print(nnp)
        sys.exit()
        arr[indx[k]] = sum(dum[ll]) / nnp

    dum = arr[indx[1] - 1: indx[1] + 1]
    ll = np.where(dum > 0)[0]
    nnp = len(ll)
    arr[indx[1]] = sum(dum[ll]) / nnp
    dum = arr[indx[nn - 2] - 1: indx[nn - 2] + 1]
    ll = np.where(dum > 0)[0]
    nnp = len(ll)
    arr[indx[nn - 2]] = sum(dum[ll]) / nnp
    if indx[0] == 0: arr[0] = arr[1] + (arr[1] - arr[2])
    if indx[0] != 0: arr[indx[0]] = (arr[indx[0] - 1] + arr[indx[0] + 1]) / 2.
    if indx[nn - 1] == nn1 - 1: arr[nn1 - 1] = arr[nn1 - 2] + (arr[nn1 - 2] - arr[nn1 - 3])
    if indx[nn - 1] != nn1 - 1: arr[indx[nn - 1]] = (arr[indx[nn - 1] - 1] + arr[indx[nn - 1] + 1]) / 2.
    return arr

def joint_arr(A, R, a):
    return A*(R/a)**2


def create_lpa(imgsz=72, tup=None):
    """
    Create the Lightcurve Parameters Array
    :param imgsz: image size
    :param tup: planetary system's parameters
    :return:
    """
    tFsq, tAg, tRp, tMp, ta, tincl, tphi0 = tup

    tjoint = joint_arr(tAg, tRp, ta)
    max_joint = Ag[-1] * (Rp[-1] / a[0]) ** 2

    # Generate LSPA
    lpa = np.zeros((imgsz, imgsz))  # initialise LSPA
    lpa[:61, :61] = tFsq  # normalised lightcurve

    lpa[61:, :12] = tAg  # geometric albedo
    lpa[:12, 61:] = tAg  # geometric albedo

    lpa[61:, 1 * 12:2 * 12] = (np.log10(tRp) - np.log10(Rp[0])) / (np.log10(Rp[-1]) - np.log10(Rp[0]))  # Planetary radius
    lpa[1 * 12:2 * 12, 61:] = (np.log10(tRp) - np.log10(Rp[0])) / (np.log10(Rp[-1]) - np.log10(Rp[0]))  # Planetary radius

    if tjoint > min_joint:
        lpa[61:, 2 * 12:3 * 12] = (np.log10(tjoint) - np.log10(min_joint)) / (np.log10(max_joint) - np.log10(min_joint))  # Joint Parameter
        lpa[2 * 12:3 * 12, 61:] = (np.log10(tjoint) - np.log10(min_joint)) / (np.log10(max_joint) - np.log10(min_joint))  # Joint Parameter
    else:
        lpa[61:, 2 * 12:3 * 12] = 0.0   # Joint Parameter
        lpa[2 * 12:3 * 12, 61:] = 0.0   # Joint Parameter

    lpa[61:, 3 * 12:4 * 12] = (np.log10(ta) - np.log10(a[0])) / (np.log10(a[-1]) - np.log10(a[0]))  # Orbital semi-major axis
    lpa[3 * 12:4 * 12, 61:] = (np.log10(ta) - np.log10(a[0])) / (np.log10(a[-1]) - np.log10(a[0]))  # Orbital semi-major axis

    lpa[61:, 4 * 12:5 * 12 + 1] = (tincl - incl[0]) / (incl[-1] - incl[0])  # Orbital inclination
    lpa[4 * 12:5 * 12 + 1, 61:] = (tincl - incl[0]) / (incl[-1] - incl[0])  # Orbital inclination

    lpa[61:, 61:] = (tphi0 - phi0[0]) / (phi0[-1] - phi0[0])  # Orbital phase
    return lpa

def unpack_from_fits(tup, rfile, planet=False, imgsz=72, kernel_size=21):
        """
                This function unpacks each fits file, extracts the file and
                it is used as input for a multithreading unpacking function ('get_training_set')
                :param tup: index for the multithreading unpacking
                :return:
                """

        mode = 'corrected'

        # Unpack the tuple
        Ag, Rp, a, incl, phi0 = tup

        if mode == 'corrected': mod = 7
        elif mode == 'raw':     mod = 3
        else:                   mod = None

        Mp = Rp * MJUP / RJUP  # DEPRECATED, the model does not use the planetary mass

        # Define the fits file and open it
        # rfile = self.file_list[idx]
        hdul = fits.open(rfile)

        # Unpack corr data
        data = hdul[1].data
        data = np.array([np.array(list(datax)) for datax in data])
        time = np.array(data[:, 0])[:imgsz ** 2]
        flux = np.array(data[:, mod])[:imgsz ** 2]

        where_are_NaNs = np.where(np.isnan(flux))
        flux[where_are_NaNs] = np.nanmean(flux)
        flux = medfilt(flux, kernel_size=kernel_size)
        
        if planet:
            # Include the planet
            mplanet = PLANET(flux, Rp=Rp, r=a, Ag=Ag, i=incl, phi0=phi0, time_arr=time)
            Fplanet = mplanet.Fp_Fs_ratio() * flux
            flux += Fplanet

        # Normalise
        flux /= np.nanmax(flux)
        max_flux = np.nanmax(flux)
        min_flux = np.nanmin(flux)
        flux = (flux - min_flux) / (max_flux - min_flux)

        Fp = np.zeros(61 ** 2)
        Fp[:len(flux)] = flux
        Fsq = Fp.reshape(61, 61)

        sflux = create_lpa(imgsz=imgsz, tup=(Fsq, Ag, Rp, Mp, a, incl, phi0))

        return sflux



def decipher_lpa(lpa_batch):
    batch_size = len(lpa_batch)

    """
    :param lpa: input Lightcurve Parameters Array
    :return: dictionary of parameters related to input lpa
    """
    param = {}
    param['albedo'] = []
    param['radius'] = []
    param['jointpar'] = []
    param['semiaxis'] = []
    param['inclination'] = []
    param['phase'] = []

    for lpa_idx in range(batch_size):
        lpa = lpa_batch[lpa_idx, :, :, 0]
        norm_albedo = np.nanmean(np.array([lpa[61:, :12], lpa[:12, 61:].T]))
        lpa_albedo = norm_albedo
        param['albedo'].append(lpa_albedo)

        norm_radius = np.nanmean(np.array([lpa[61:, 1 * 12:2 * 12], lpa[1 * 12:2 * 12, 61:].T]))
        lpa_radius = (norm_radius * (np.log10(Rp[-1]) - np.log10(Rp[0]))) + np.log10(Rp[0])
        lpa_radius = 10**lpa_radius
        param['radius'].append(lpa_radius)

        max_joint = Ag[-1] * (Rp[-1] / a[0]) ** 2

        norm_joint = np.nanmean(np.array([lpa[61:, 2 * 12:3 * 12], lpa[2 * 12:3 * 12, 61:].T]))
        if norm_joint > 0:
            lpa_joint = (norm_joint * (np.log10(max_joint) - np.log10(min_joint))) + np.log10(min_joint)
            lpa_joint = 10**lpa_joint
        else:
            lpa_joint = 0.0
        param['jointpar'].append(lpa_joint)

        norm_semiaxis = np.nanmean(np.array([lpa[61:, 3 * 12:4 * 12], lpa[3 * 12:4 * 12, 61:].T]))
        lpa_semiaxis = (norm_semiaxis * (np.log10(a[-1]) - np.log10(a[0]))) + np.log10(a[0])
        lpa_semiaxis = 10**lpa_semiaxis
        param['semiaxis'].append(lpa_semiaxis)

        norm_inclination = np.nanmean(np.array([lpa[61:, 4 * 12:5 * 12 + 1], lpa[4 * 12:5 * 12 + 1, 61:].T]))
        lpa_inclination = (norm_inclination * (incl[-1] - incl[0])) + incl[0]
        param['inclination'].append(lpa_inclination)

        norm_phase = np.nanmean(np.array(lpa[61:, 61:]))
        lpa_phase = (norm_phase * (phi0[-1] - phi0[0])) + phi0[0]
        param['phase'].append(lpa_phase)

    return param

def plot_histogram(deciphered, label, idx, outdir=''):
    deciphered[label] = np.array(deciphered[label])
    if label == 'albedo':
        plt.title('Albedo')
        plt.xlabel('Albedo')
        var = Ag
        div = 1
        hist = deciphered[label] / div
        bins = np.linspace(min(var), max(var), 32)/div

    elif label == 'radius':
        plt.title('Radius')
        plt.xlabel('$R_p(R_{JUP}$)')
        plt.xscale('log')
        var = Rp
        div = RJUP
        hist = deciphered[label] / div
        bins = np.logspace(min(np.log10(var)), max(np.log10(var)), 32)/div
    elif label == 'jointpar':
        plt.title('Joint Parameter')
        plt.xlabel(r'$A \frac{R_p^2}{a^2}$')
        plt.xscale('log')
        var = 1
        div = 1
        max_joint = Ag[-1] * (Rp[-1] / a[0]) ** 2
        hist = deciphered[label]/div
        bins = np.logspace(np.log10(min_joint), np.log10(max_joint), 32)/div
    elif label == 'semiaxis':
        plt.title('Semi-major Axis')
        plt.xlabel("$a$(AU)")
        plt.xscale('log')
        var = a
        div = AU
        hist = deciphered[label] / div
        bins = np.logspace(min(np.log10(var)), max(np.log10(var)), 32)/div
    elif label == 'inclination':
        plt.title('Inclination')
        plt.xlabel('i')
        var = incl
        div = 1
        hist = deciphered[label] / div
        bins = np.linspace(min(var), max(var), 32) / div
    elif label == 'phase':
        plt.title('Initial Phase')
        plt.xlabel(r'$\varphi_0$')
        var = phi0
        div = 1
        hist = deciphered[label] / div
        bins = np.linspace(min(var), max(var), 32) / div

    if label != "truths":
        make_dir(os.path.join(outdir,'histograms/%s' % label))

        truth = deciphered['truths'][label][0] / div
        weights = np.ones_like(hist) / len(hist)
        plt.hist(hist,
                 color='olivedrab',
                 bins=bins,
                 weights=weights,
                 edgecolor='k',
                 linewidth=1.0)
        plt.plot([truth, truth], [0, 1], c='orange', label='truth')
        plt.legend()
        plt.ylim(0, 1)
        plt.savefig(str(os.path.join(outdir, "histograms/%s" % label)) + "/%s_%04d.pdf" % (label, idx))
        plt.close()

def plot_existing_planets(label, planet, ax):
    if "k2-93" in planet:
        inclinations = np.array([88.4, 89.58]) * np.pi / 180 + np.pi / 2
        semiaxii = np.array([19.5, 73]) * (1.4 * RSOL) / AU
        radii = np.array([0.0188, 0.0166]) * (1.4 *RSOL) / RJUP
        colors = ['red', 'violet']
        labels = ["HIP 41378 b", "HIP 41378 c"]

    elif "k2-100" in planet:
        inclinations = np.array([85.1]) * np.pi / 180 + np.pi / 2
        semiaxii = np.array([0.0292])
        radii = np.array([0.31])
        colors = ['red']
        labels = ["K2 100 b"]

    elif "k2-184" in planet:
        inclinations = np.array([89.47]) * np.pi / 180 + np.pi / 2
        semiaxii = np.array([0.124541])
        radii = np.array([0.1374])
        colors = ['red']
        labels = ["K2 184 b"]

    elif "k2-236" in planet:
        inclinations = np.array([87.9]) * np.pi / 180 + np.pi / 2
        semiaxii = np.array([0.148])
        radii = np.array([0.546])
        colors = ['red']
        labels = ["K2 236 b"]

    if label == "inclination":
        for ii in range(len(inclinations)):
            ax.plot([inclinations[ii], inclinations[ii]], [0, 1], c=colors[ii], linewidth=2, label=labels[ii])
    if label == "semiaxis":
        for ii in range(len(semiaxii)):
            ax.plot([semiaxii[ii], semiaxii[ii]], [0, 1], c=colors[ii], linewidth=2, label=labels[ii])
    if label == "radius":
        for ii in range(len(radii)):
            ax.plot([radii[ii], radii[ii]], [0, 1], c=colors[ii], linewidth=2, label=labels[ii])


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

def prepare_test(file_paths):
    """
    Returns a list of all the lightcurves array in the given directory
    :param root: dataset directory
    :return: list of arrays
    """
    try:
        file_paths = glob(file_paths + "*.pgz")
    except TypeError:
        pass
    training_array = []
    for file_path in file_paths:
        if file_path[-3:] == 'pgz':
            arr_list = pickle.load(gzip.open(file_path, 'rb'))
            for arr in arr_list:
                arr = arr.reshape(72, 72)
                training_array.append(arr)
        elif file_path[-3:] == "npy":
            arr = np.load(file_path)
            arr = arr.reshape(72, 72)
            training_array.append(arr)
    training_array = np.array(training_array)
    return training_array

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

def file_is_modified(filePath, logfile):
    """
    Check wheter a file has been modified, this function is particularly
    to check the library_grid file and generate the tuple of parameters to be run
    and simulate the appropriate planetary systems
    :param filePath: file to check
    :return: Boolean: returns a variable to indicate whether the file is the same
    of last run of it has been modified
    """

    fileStatsObj = os.stat(filePath)

    modificationTime = time.ctime(fileStatsObj[stat.ST_MTIME])

    # logging.info("Last Modified Time for '%s': " % filePath.split("/")[-1], modificationTime)

    logfile = {}
    logfile['modifications'] = {}
    logfile['modifications'][filePath] = modificationTime

    if not os.path.exists("log.pickle"):
        pickle.dump(logfile, open("log.pickle", "wb"))
        logging.info("log.pickle file has been created")
        return True
    else:
        last_modified = pickle.load(open("log.pickle", "rb"))
        check = last_modified['modifications'][filePath] # == modificationTime
        if not check:
            logging.info("%s has been modified!" % filePath)
            pickle.dump(logfile, open("log.pickle", "wb"))
            return True
        else:
            logging.info("%s has always been the same!" % filePath)
            return False

def create_parameters_tuple(file_list):
    file_list = glob.glob(file_list)

    import libraries.library_grid as lg
    """
    Create a tuple to allow multiprocessing generation of dataset
    """
    logging.info("Creation of parameters' tuple")
    all_grid = []
    # Tuple creation for planetary grid
    for x0 in range(len(file_list)):
        for x1 in lg.Ag:
            for x2 in lg.Rp:
                for x3 in lg.a:
                    for x4 in lg.incl:
                        for x5 in lg.phi0:
                            all_grid.append((int(x0), x1, x2, x3, x4, x5))
    all_grid = np.array(all_grid)
    np.random.shuffle(all_grid)
    return all_grid
    # np.save("training_tuples.npy", all_grid)

def download_server(server_name='curta.mcia.fr', 
                    username='tzingales', 
                    check_fold='/gpfs/home/tzingales/repositories/pythonpackages/ExoGazer/exogazer/tests/0/',
                    main_fold = "/gpfs/home/tzingales/repositories/pythonpackages/ExoGazer/exogazer/",
                    my_fold = "/Users/tiziano/Dropbox/PycharmProjects/pypackages/ExoGazer/exogazer/"):
    def get_remote_list(check_fold):
        import paramiko
        import os

        plan_simul = []
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server_name, username=username)

        # check_fold = "/gpfs/home/tzingales/repositories/DeepStar/tests/0"
        rawcommand = 'ls %s' % check_fold
        command = rawcommand.format(path=check_fold)
        stdin, stdout, stderr = ssh.exec_command(command)
        filelist = stdout.read().splitlines()

        ftp = ssh.open_sftp()

        for afile in filelist:
            (head, filename) = os.path.split(afile)
            filename = str(filename)
            filename = filename[2:-1] + "/"
            plan_simul.append(filename)
        ftp.close()
        ssh.close()
        return plan_simul

    Nsimul = 10

    # main_fold = "/gpfs/home/tzingales/repositories/DeepStar/"
    # my_fold = "/Users/tiziano/Dropbox/PycharmProjects/DeepStar/"

    plan_simul = get_remote_list(check_fold)
    print("Checking remote list...")
    pbar = tqdm(total=len(plan_simul))
    for ii in range(len(plan_simul)):
        k2_name = get_remote_list(check_fold+plan_simul[ii])
        # print(plan_simul[ii]+k2_name[0])
        plan_simul[ii] += k2_name[0]
        pbar.update()
    pbar.close()
    # print(plan_simul[:2])
    # exit(0)
    my_simul = os.listdir(my_fold + "tests/0/")

    plan_simul = [cob_file for cob_file in plan_simul if cob_file[:-1] not in my_simul]
    
    print("Downloading planetary analysis...")
    pbar = tqdm(total=int(len(plan_simul)*Nsimul))
    for ii in range(Nsimul):
        for jj in range(len(plan_simul)):
            cob_file = main_fold + "tests/%d/" % ii + plan_simul[jj] + "histograms/output.pickle"
            my_dir = my_fold + "tests/%d/" % ii + plan_simul[jj] + "histograms/"
            make_dir(my_dir)
            os.system("scp %s@%s:%s" % (username, server_name, cob_file) + " %s" % my_dir + " >/dev/null 2>&1")
            pbar.update()
    pbar.close()

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
    return scipy.misc.imread(path, mode='RGB').astype(np.float)


def merge_images(images, size):
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


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [8, 8], './samples/test_%s.png' % strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [8, 8], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)
