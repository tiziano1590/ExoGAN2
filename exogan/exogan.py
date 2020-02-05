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
    
    dataset_fits_files = genpars['dataset_fits_files']
    if dataset_fits_files[-1] != '/':
        dataset_fits_files += '/'
    
    if generate_training_set:
        # first of all check that the parameters grid has not been modified,
        # it is important to normalise the dataset and then decode all the
        # network outputs
        # if file_is_modified(genpars['library_file'], genpars['log_file']):
        all_grid = create_parameters_tuple(dataset_fits_files+"*.fits")
    
        # load all possible combinations for parameters array
        # it is saved in an external file to simplify the
        # parallelisation between different nodes
        # logging.info("Load training parameters")
        # all_grid = np.load("training_tuples.npy")
    
        # split dataset to run in different servers
        min_lim = int(node * int(len(all_grid)/nodes_number))
        max_lim = int(node + 1 * int(len(all_grid)/nodes_number))
    
    
        file_list = glob.glob(dataset_fits_files+"*.fits")
        data = DATA(all_grid=all_grid,
                    min_lim=min_lim,
                    max_lim=max_lim,
                    file_list=file_list,
                    nthreads=nthreads,
                    planet=planet)
        corr_training_set, corr_test_set = data.get_training_set(mode='corrected')
        
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
        
        test_out_directory = comppars['save_synth_planets_out_info']
        if test_out_directory[-1] != '/': test_out_directory += '/'
        
        test_set_dir = comppars['test_set_dir']
        if test_set_dir[-1] != '/': test_set_dir += '/'
        
        imgSize = int(comppars['imgSize'])
        lam = comppars['lam']
                
        test_set = glob.glob(test_set_dir + "*.fits")

    print('ExoGAN PROGRAM FINISHES AT %s' % datetime.datetime.now())

if __name__ == "__main__":
    main()