API Reference
=============

This section provides detailed documentation for all ExoGAN 2 modules and their functions.

Main Module
-----------

.. automodule:: exogan.exogan
   :members:
   :undoc-members:
   :show-inheritance:

The main entry point for the ExoGAN 2 command-line interface. Handles argument parsing and coordinates training and completion workflows.

Model Module
------------

.. automodule:: exogan.model.model
   :members:
   :undoc-members:
   :show-inheritance:

**DCGAN Class**

The core DCGAN (Deep Convolutional Generative Adversarial Network) implementation for atmospheric spectrum generation and completion.

**Key Methods:**

- ``__init__(sess, genpars)``: Initialize the DCGAN model
- ``build_model()``: Construct the TensorFlow computational graph
- ``train(trainpars)``: Train the generator and discriminator networks
- ``complete(comppars, spectrum, parfile)``: Perform atmospheric retrieval
- ``generator(z)``: Generator network architecture
- ``discriminator(image, reuse=False)``: Discriminator network architecture

Parameter Module
----------------

.. automodule:: exogan.parameter.parameterparser
   :members:
   :undoc-members:
   :show-inheritance:

**ParameterParser Class**

Handles reading and parsing configuration files for ExoGAN 2.

**Key Methods:**

- ``read(filename)``: Load parameters from configuration file
- ``generalpars()``: Extract general model parameters
- ``trainpars()``: Extract training-specific parameters
- ``comppars()``: Extract completion-specific parameters

Libraries Module
----------------

.. automodule:: exogan.libraries.libraries
   :members:
   :undoc-members:
   :show-inheritance:

**Grids Class**

Manages wavelength and wavenumber grids for spectral data processing.

**Attributes:**

- ``wnw_grid``: Wavenumber grid data loaded from file

Utilities Module
----------------

.. automodule:: exogan.util.utils
   :members:
   :undoc-members:
   :show-inheritance:

Provides utility functions for data processing, visualization, and I/O operations.

**Key Functions:**

- ``get_spectral_matrix(spectrum, parfile)``: Convert spectrum to matrix format
- ``get_aspa_dataset_from_hdf5(train_path, num_chunks)``: Load training data from HDF5
- ``load_dict_from_hdf5(filename)``: Load dictionary from HDF5 file
- ``save_dict_to_hdf5(dic, filename)``: Save dictionary to HDF5 file
- ``make_corner_plot(all_hists, ranges, labels, ground_truths, comppars, index)``: Generate corner plots
- ``directory(path)``: Process directory paths
- ``make_dir(dirname, comppars)``: Create directories with proper structure

Tools Module
------------

.. automodule:: exogan.tools.tools
   :members:
   :undoc-members:
   :show-inheritance:

Additional tools and helper functions for ExoGAN 2 workflows.

Module Structure
----------------

ExoGAN 2 is organized into the following packages:

.. code-block:: text

   exogan/
   ├── __init__.py
   ├── exogan.py          # Main CLI entry point
   ├── model/
   │   ├── __init__.py
   │   └── model.py       # DCGAN implementation
   ├── parameter/
   │   ├── __init__.py
   │   └── parameterparser.py  # Configuration parser
   ├── libraries/
   │   ├── __init__.py
   │   └── libraries.py   # Wavelength grids
   ├── util/
   │   ├── __init__.py
   │   └── utils.py       # Utility functions
   └── tools/
       ├── __init__.py
       └── tools.py       # Additional tools

Example Usage
-------------

**Using the Model Programmatically:**

.. code-block:: python

   import tensorflow as tf
   from exogan.model import DCGAN
   from exogan.parameter import ParameterParser
   
   # Load configuration
   pp = ParameterParser()
   pp.read('config.par')
   genpars = pp.generalpars()
   
   # Initialize TensorFlow session
   config = tf.compat.v1.ConfigProto()
   config.gpu_options.allow_growth = True
   
   with tf.compat.v1.Session(config=config) as sess:
       # Create DCGAN model
       model = DCGAN(sess, genpars)
       
       # Train or use the model
       trainpars = pp.trainpars()
       model.train(trainpars)

**Loading and Processing Data:**

.. code-block:: python

   from exogan.util import get_aspa_dataset_from_hdf5, get_spectral_matrix
   import numpy as np
   
   # Load training dataset
   data = get_aspa_dataset_from_hdf5('/path/to/data', num_chunks=10)
   
   # Process a single spectrum
   spectrum = np.genfromtxt('spectrum.dat')[:, 1]
   matrix = get_spectral_matrix(spectrum)

**Working with Parameters:**

.. code-block:: python

   from exogan.parameter import ParameterParser
   
   pp = ParameterParser()
   pp.read('config.par')
   
   # Access specific parameters
   batch_size = pp.generalpars()['batch_size']
   learning_rate = pp.trainpars()['learning_rate']
   
   # Get completion parameters
   comp = pp.comppars()
   input_spectrum = comp['input']
