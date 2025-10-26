Usage Guide
===========

Command Line Interface
----------------------

ExoGAN 2 provides a command-line interface for easy interaction.

Basic Syntax
^^^^^^^^^^^^

.. code-block:: bash

   exogan -i <config_file> [--train] [--completion]

Arguments
^^^^^^^^^

``-i, --input`` (required)
   Path to the configuration parameter file

``-T, --train``
   Run training mode

``-C, --completion``
   Run completion/retrieval mode

Training Workflow
-----------------

Data Preparation
^^^^^^^^^^^^^^^^

Prepare your training data as HDF5 files containing atmospheric spectra. The expected format:

- Directory with multiple ``.h5`` files
- Each file contains spectral data as matrices
- Consistent dimensions across all files

Configuration
^^^^^^^^^^^^^

Set up your training parameters in the configuration file:

.. code-block:: ini

   [Training]
   epoch = 100
   learning_rate = 0.0002
   beta1 = 0.5
   dataset = /path/to/data
   checkpoint_dir = ./checkpoints
   sample_dir = ./samples
   log_dir = ./logs

Running Training
^^^^^^^^^^^^^^^^

.. code-block:: bash

   exogan -i config.par --train

**Monitoring Training:**

- Check ``log_dir`` for TensorBoard logs
- Sample outputs are saved to ``sample_dir``
- Model checkpoints are saved to ``checkpoint_dir``

Completion Workflow
-------------------

Input Data
^^^^^^^^^^

Prepare your observed spectrum:

- Text file format: wavelength and flux columns
- Or HDF5 format matching training data structure

Configuration
^^^^^^^^^^^^^

.. code-block:: ini

   [Completion]
   checkpointDir = ./checkpoints
   lam = 0.1
   imgSize = 33
   input = spectrum.dat
   parfile = parameters.par

Running Completion
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   exogan -i config.par --completion

**Output:**

- Retrieved atmospheric parameters
- Completed spectrum
- Corner plots showing parameter distributions
- Comparison with input spectrum

Advanced Usage
--------------

Custom Architecture
^^^^^^^^^^^^^^^^^^^

Modify the DCGAN architecture by adjusting parameters:

.. code-block:: ini

   [General]
   z_dim = 100        # Latent space dimension
   gf_dim = 64        # Generator feature maps
   df_dim = 64        # Discriminator feature maps
   gfc_dim = 1024     # Generator fully connected
   dfc_dim = 1024     # Discriminator fully connected

GPU Configuration
^^^^^^^^^^^^^^^^^

For multi-GPU setups or specific GPU allocation:

.. code-block:: python

   import tensorflow as tf
   from exogan.model import DCGAN
   
   config = tf.compat.v1.ConfigProto()
   config.gpu_options.allow_growth = True
   config.gpu_options.visible_device_list = "0,1"  # Use GPUs 0 and 1
   
   with tf.compat.v1.Session(config=config) as sess:
       dcgan = DCGAN(sess, genpars)
       dcgan.train(trainpars)

Batch Processing
^^^^^^^^^^^^^^^^

Process multiple spectra in batch:

.. code-block:: python

   from exogan.parameter import ParameterParser
   import glob
   
   pp = ParameterParser()
   pp.read('config.par')
   
   spectra_files = glob.glob('data/*.dat')
   for spectrum_file in spectra_files:
       # Update configuration for each spectrum
       pp.set('Completion', 'input', spectrum_file)
       # Run completion
       # ... (completion code)

Best Practices
--------------

Training
^^^^^^^^

1. **Use adequate training data**: Minimum 10,000 spectra recommended
2. **Monitor convergence**: Check discriminator and generator losses
3. **Regular checkpointing**: Save models every few epochs
4. **Validation**: Keep a separate validation set

Completion
^^^^^^^^^^

1. **Check data quality**: Ensure input spectrum matches training data format
2. **Tune lambda parameter**: Balance between contextual and perceptual loss
3. **Multiple runs**: Run completion multiple times to assess uncertainty
4. **Visual inspection**: Always inspect generated spectra visually

Troubleshooting
---------------

Training Issues
^^^^^^^^^^^^^^^

**Mode collapse:**
   - Reduce learning rate
   - Adjust batch size
   - Add noise to discriminator inputs

**Slow convergence:**
   - Increase learning rate
   - Use GPU acceleration
   - Reduce model complexity

Completion Issues
^^^^^^^^^^^^^^^^^

**Poor retrieval quality:**
   - Check that trained model converged properly
   - Adjust lambda parameter
   - Verify input data format matches training data

**Out of memory errors:**
   - Reduce batch size
   - Enable memory growth in GPU config
   - Process fewer spectra simultaneously
