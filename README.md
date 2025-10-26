# ExoGAN 2

ExoGAN 2 is an atmospheric analysis framework for exoplanet characterization using Generative Adversarial Networks (GANs).

## Installation

Install ExoGAN 2 using pip:

```bash
pip install git+https://github.com/tiziano1590/ExoGAN2.git
```

Or install from source:

```bash
git clone https://github.com/tiziano1590/ExoGAN2.git
cd ExoGAN2
pip install -e .
```

## Usage

ExoGAN 2 provides a command-line interface for training and completion:

### Training Mode

```bash
exogan -i config.par --train
```

### Completion Mode

```bash
exogan -i config.par --completion
```

## Documentation

Full documentation is available at: https://tiziano1590.github.io/ExoGAN2/

## Requirements

- Python 3.8 - 3.11 (recommended: 3.11)
- TensorFlow >= 2.0
- NumPy
- SciPy
- Matplotlib
- h5py
- Pandas
- Astropy
- And more (see pyproject.toml)

## Author

Tiziano Zingales - Università degli Studi di Padova

## License

MIT License - see [LICENSE](LICENSE) file for details
