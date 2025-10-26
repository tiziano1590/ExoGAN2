ExoGAN 2 Documentation
======================

ExoGAN 2 is an atmospheric analysis framework for exoplanet characterization using Generative Adversarial Networks (GANs).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Installation
============

Install ExoGAN 2 using pip:

.. code-block:: bash

   pip install exogan

Or install from source:

.. code-block:: bash

   git clone https://github.com/tiziano1590/ExoGAN2.git
   cd ExoGAN2
   pip install -e .

Usage
=====

Command Line Interface
----------------------

ExoGAN 2 provides a command-line interface for training and completion:

Training Mode
~~~~~~~~~~~~~

.. code-block:: bash

   exogan -i config.par --train

Completion Mode
~~~~~~~~~~~~~~~

.. code-block:: bash

   exogan -i config.par --completion

API Reference
=============

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
