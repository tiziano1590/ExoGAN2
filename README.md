<div align="center">

# 🌟 ExoGAN 2

### Atmospheric Analysis Framework for Exoplanet Characterization

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://tiziano1590.github.io/ExoGAN2/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%3E%3D2.0-orange.svg)](https://www.tensorflow.org/)

[Documentation](https://tiziano1590.github.io/ExoGAN2/) • [Installation](#-installation) • [Usage](#-usage) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

---

## 🔭 Overview

**ExoGAN 2** is a powerful deep learning framework for analyzing exoplanetary atmospheres using Generative Adversarial Networks (GANs). Built on TensorFlow, it enables researchers to perform atmospheric retrieval and parameter estimation from spectroscopic observations.

### ✨ Key Features

- 🧠 **Deep Learning Powered**: Utilizes DCGAN (Deep Convolutional GAN) architecture for spectrum analysis
- ⚡ **GPU Accelerated**: Optimized for high-performance computation with CUDA support
- 🔬 **Scientific Accuracy**: Designed for rigorous exoplanet atmospheric characterization
- 🎯 **Flexible Architecture**: Modular design allows easy customization and extension
- 📊 **Comprehensive Analysis**: Complete toolkit for training, testing, and atmospheric retrieval
- 🌐 **Well Documented**: Extensive documentation with examples and API references

### 🎯 What Can ExoGAN 2 Do?

- Train custom models on atmospheric spectra datasets
- Perform atmospheric parameter retrieval from observed spectra
- Generate synthetic spectra for validation and testing
- Analyze spectral features and molecular abundances
- Visualize results with corner plots and parameter distributions

---

## 📦 Installation

### Quick Install

```bash
pip install git+https://github.com/tiziano1590/ExoGAN2.git
```

### From Source

For development or latest features:

```bash
git clone https://github.com/tiziano1590/ExoGAN2.git
cd ExoGAN2
pip install -e .
```

### Using Conda

Create a dedicated environment:

```bash
conda env create -f environment.yml
conda activate exogan
pip install -e .
```

### System Requirements

- **Python**: 3.8 - 3.11 (3.11 recommended)
- **OS**: Linux, macOS, Windows
- **GPU**: CUDA-compatible GPU recommended for training

---

## 🚀 Quick Start

### Training a Model

Create a configuration file `config.par`:

```ini
[General]
batch_size = 64
image_size = 33
z_dim = 100

[Training]
epoch = 100
learning_rate = 0.0002
dataset = /path/to/training/data
checkpoint_dir = ./checkpoints
```

Train the model:

```bash
exogan -i config.par --train
```

### Atmospheric Retrieval

Run atmospheric completion on observed spectra:

```bash
exogan -i config.par --completion
```

### Python API

Use ExoGAN 2 programmatically:

```python
import tensorflow as tf
from exogan.parameter import ParameterParser
from exogan.model import DCGAN

# Load configuration
pp = ParameterParser()
pp.read('config.par')
genpars = pp.generalpars()

# Initialize and train model
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    dcgan = DCGAN(sess, genpars)
    trainpars = pp.trainpars()
    dcgan.train(trainpars)
```

---

## 📚 Examples

### Basic Workflow

1. **Prepare your data**: Organize spectral data in HDF5 format
2. **Configure parameters**: Create a `.par` configuration file
3. **Train the model**: Run training mode with your dataset
4. **Retrieve parameters**: Use completion mode on observed spectra
5. **Analyze results**: Examine corner plots and parameter distributions

### Example Configuration

See the [documentation](https://tiziano1590.github.io/ExoGAN2/quickstart.html) for complete configuration examples.

---

## 📖 Documentation

Comprehensive documentation is available at **[tiziano1590.github.io/ExoGAN2](https://tiziano1590.github.io/ExoGAN2/)**

- 📘 [Installation Guide](https://tiziano1590.github.io/ExoGAN2/installation.html)
- 🚀 [Quick Start Tutorial](https://tiziano1590.github.io/ExoGAN2/quickstart.html)
- 📖 [Usage Guide](https://tiziano1590.github.io/ExoGAN2/usage.html)
- 🔧 [API Reference](https://tiziano1590.github.io/ExoGAN2/api_modules.html)

---

## 🛠️ Core Dependencies

- **TensorFlow** ≥ 2.0 - Deep learning framework
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing
- **Matplotlib** - Visualization
- **h5py** - HDF5 data handling
- **Pandas** - Data manipulation
- **Astropy** - Astronomical computations
- **corner** - Corner plot generation

For a complete list, see [`pyproject.toml`](pyproject.toml).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

For major changes, please open an issue first to discuss proposed changes.

---

## 📝 Citation

If you use ExoGAN 2 in your research, please cite:

```bibtex
@software{exogan2,
  author = {Zingales, Tiziano},
  title = {ExoGAN 2: Atmospheric Analysis Framework for Exoplanet Characterization},
  year = {2025},
  url = {https://github.com/tiziano1590/ExoGAN2},
  institution = {Università degli Studi di Padova}
}
```

---

## 👤 Author

**Tiziano Zingales**  
Università degli Studi di Padova  
📧 tiziano.zingales@unipd.it

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

ExoGAN 2 builds upon advances in deep learning and exoplanet science. We acknowledge the contributions of the broader scientific community in developing the tools and techniques that make this work possible.

---

<div align="center">

**[⬆ Back to Top](#-exogan-2)**

Made with ❤️ for the exoplanet community

</div>
