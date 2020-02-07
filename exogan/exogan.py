"""
Created on Fri Dec 6 17:04:55 2019

The main ExoGazer program

@author: Tiziano Zingales - Université de Bordeaux
"""

def main():
    import argparse
    import datetime
    import tensorflow as tf
    from exogan.parameter import ParameterParser
    from exogan.util import directory, make_dir
    import numpy as np
    import glob
    import matplotlib.pyplot as plt
    import pickle
    from exogan.model import DCGAN
    import os
    
    parser = argparse.ArgumentParser(description='ExoGazer')
    
    parser.add_argument("-i", "--input", dest='input_file', type=str,
                        required=True, help="Input par file to pass")

    parser.add_argument("-T", "--train", dest='train', default=False,
                        help="When set, runs the training phase", action='store_true')
    
    parser.add_argument("-C", "--completion", dest='completion', default=False,
                        help="When set, runs the completion phase", action='store_true')

    args = parser.parse_args()
    
    print('ExoGAN PROGRAM STARTS AT %s' % datetime.datetime.now())
    
    # Parse the input file
    pp = ParameterParser()
    pp.read(args.input_file)
    genpars = pp.generalpars()

    if args.train:

        trainpars = pp.trainpars()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            dcgan = DCGAN(sess, genpars)
            dcgan.train(trainpars)
            
    if args.completion:
        comppars = pp.comppars()
        checkpointDir = comppars['checkpointDir']
        check_range = comppars['checkpointsRange']
        planet_prob = comppars['synth_planet_probability']
        checkpointsRange = [checkpointDir+str(ii) for ii in range(int(check_range[0]), int(check_range[1])+1)]


    print('ExoGAN PROGRAM FINISHES AT %s' % datetime.datetime.now())

if __name__ == "__main__":
    main()