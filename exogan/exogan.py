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
    if genpars["suppress_tf_warnings"]:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
    trainpars = pp.trainpars()
    
    nthreads = int(genpars['gen_trainset_nthreads'])
    data_type = genpars['data_type']
    planet = genpars['planet']
    generate_training_set = genpars['generate_training_set']
    train_dcgan = genpars['train_dcgan']
    nodes_number = int(genpars['nodes_number'])
    node = int(genpars['node'])

    if args.train:
        
        batch_size =     int(trainpars['batch_size'])
        image_size =     int(trainpars['image_size'])
        checkpoint_dir = str(trainpars['checkpoint_dir'])
        sample_dir =     str(trainpars['sample_dir'])
        log_dir =        str(trainpars['log_dir'])
        
        make_dir(checkpoint_dir)
        make_dir(sample_dir)
        make_dir(log_dir)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            dcgan = DCGAN(sess, image_size=image_size, batch_size=batch_size,
                          is_crop=False, checkpoint_dir=checkpoint_dir)
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