Installation
============

Requirements
------------

ExoGAN 2 requires Python 3.8 or newer (Python 3.11 recommended).

**Core Dependencies:**

- TensorFlow >= 2.0
- NumPy
- SciPy
- Matplotlib
- h5py
- Pandas
- Astropy
- configobj
- tqdm
- Pillow
- corner
- imageio

Installation Methods
--------------------

Using pip
^^^^^^^^^

The simplest way to install ExoGAN 2:

.. code-block:: bash

   pip install git+https://github.com/tiziano1590/ExoGAN2.git

From Source
^^^^^^^^^^^

For development or to get the latest changes:

.. code-block:: bash

   git clone https://github.com/tiziano1590/ExoGAN2.git
   cd ExoGAN2
   pip install -e .

The ``-e`` flag installs the package in editable mode, allowing you to modify the source code.

Using Conda
^^^^^^^^^^^

Create a conda environment using the provided environment file:

.. code-block:: bash

   # Create environment
   conda env create -f environment.yml
   
   # Activate environment
   conda activate exogan
   
   # Install ExoGAN 2
   pip install -e .

Verification
------------

Verify your installation:

.. code-block:: bash

   exogan --help

Or in Python:

.. code-block:: python

   import exogan
   from exogan.model import DCGAN
   print("ExoGAN 2 installed successfully!")

Troubleshooting
---------------

TensorFlow GPU Support
^^^^^^^^^^^^^^^^^^^^^^

For GPU acceleration, ensure you have:

- CUDA-compatible GPU
- CUDA Toolkit installed
- cuDNN library installed

Refer to the `TensorFlow GPU guide <https://www.tensorflow.org/install/gpu>`_ for detailed instructions.

Common Issues
^^^^^^^^^^^^^

**ImportError: No module named 'tensorflow'**

Install TensorFlow:

.. code-block:: bash

   pip install tensorflow

**Version Conflicts**

If you encounter version conflicts, try creating a fresh environment:

.. code-block:: bash

   conda create -n exogan python=3.11
   conda activate exogan
   pip install git+https://github.com/tiziano1590/ExoGAN2.git
