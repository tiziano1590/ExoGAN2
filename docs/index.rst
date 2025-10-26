ExoGAN 2 Documentation
======================

.. image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://github.com/tiziano1590/ExoGAN2/blob/main/LICENSE
   :alt: License

Welcome to ExoGAN 2's documentation!

ExoGAN 2 is a powerful atmospheric analysis framework for exoplanet characterization using Generative Adversarial Networks (GANs). This framework enables researchers to analyze and retrieve atmospheric parameters from exoplanet spectra using deep learning techniques.

Key Features
------------

- **Deep Learning Based**: Utilizes DCGAN (Deep Convolutional Generative Adversarial Networks) for spectrum analysis
- **High Performance**: Optimized for GPU acceleration with TensorFlow  
- **Flexible Architecture**: Modular design allows easy customization and extension
- **Comprehensive Tools**: Complete toolkit for training, testing, and atmospheric retrieval
- **Scientific Accuracy**: Designed for rigorous scientific analysis of exoplanetary atmospheres

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install git+https://github.com/tiziano1590/ExoGAN2.git

Basic Usage
^^^^^^^^^^^

Training a model:

.. code-block:: bash

   exogan -i config.par --train

Running atmospheric completion:

.. code-block:: bash

   exogan -i config.par --completion

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
