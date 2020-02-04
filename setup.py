from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ExoGAN',
    url='https://github.com/ucl-exoplanets/ExoGAN2.git',
    author='Tiziano Zingales',
    author_email='tiziano.zingales@u-bordeaux.fr',
    # Needed to actually package something
    packages=['exogan'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='2.0',
    # The license can be anything you like
    license='LAB',
    description='Deep Convolutional Neural Network to study exoplanetary spectra',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.md').read(),
)
