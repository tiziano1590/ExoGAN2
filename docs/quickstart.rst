Quick Start Guide
=================

This guide will help you get started with ExoGAN 2 for exoplanet atmospheric analysis.

Basic Workflow
--------------

The typical ExoGAN 2 workflow consists of:

1. **Prepare Configuration** - Create a parameter file with your settings
2. **Train Model** - Train the DCGAN on your dataset (optional if using pre-trained model)
3. **Run Completion** - Perform atmospheric retrieval on observed spectra

Configuration File
------------------

ExoGAN 2 uses parameter files (``.par`` format) for configuration. Here's a basic example:

.. code-block:: ini

   [General]
   batch_size = 64
   image_size = 33
   sample_size = 64
   c_dim = 1
   z_dim = 100
   gf_dim = 64
   df_dim = 64
   gfc_dim = 1024
   dfc_dim = 1024
   lam = 0.1
   is_crop = False
   
   [Training]
   epoch = 100
   learning_rate = 0.0002
   beta1 = 0.5
   train_size = inf
   batch_size = 64
   dataset = /path/to/training/data
   checkpoint_dir = ./checkpoints
   sample_dir = ./samples
   log_dir = ./logs
   training_set_ratio = 1.0
   num_chunks = 10
   
   [Completion]
   checkpointDir = ./checkpoints
   lam = 0.1
   imgSize = 33
   input = /path/to/spectrum.dat
   parfile = /path/to/parameters.par

Training Mode
-------------

To train a new model from scratch:

.. code-block:: bash

   exogan -i config.par --train

This will:

1. Load training data from the specified dataset directory
2. Initialize the DCGAN architecture
3. Train the discriminator and generator networks alternately
4. Save checkpoints periodically to the checkpoint directory
5. Generate sample outputs during training

**Training Tips:**

- Use GPU for faster training (highly recommended)
- Monitor the training logs to check for convergence
- Adjust learning rate if training is unstable
- Use a larger dataset for better results

Completion Mode
---------------

To perform atmospheric retrieval on observed spectra:

.. code-block:: bash

   exogan -i config.par --completion

This mode:

1. Loads a trained model from the checkpoint directory
2. Reads the input spectrum
3. Performs optimization in the latent space
4. Generates the completed spectrum with retrieved parameters

Python API Example
------------------

You can also use ExoGAN 2 programmatically in Python:

.. code-block:: python

   import tensorflow as tf
   from exogan.parameter import ParameterParser
   from exogan.model import DCGAN
   
   # Load configuration
   pp = ParameterParser()
   pp.read('config.par')
   genpars = pp.generalpars()
   trainpars = pp.trainpars()
   
   # Create TensorFlow session
   config = tf.compat.v1.ConfigProto()
   config.gpu_options.allow_growth = True
   
   with tf.compat.v1.Session(config=config) as sess:
       # Initialize model
       dcgan = DCGAN(sess, genpars)
       
       # Train the model
       dcgan.train(trainpars)

Next Steps
----------

- Read the :doc:`usage` guide for detailed information
- Explore the :doc:`api_modules` for API documentation
- Check example parameter files in the repository
