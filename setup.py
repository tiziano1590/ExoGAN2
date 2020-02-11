#!/usr/bin/env python
from setuptools import find_packages
from numpy.distutils.core import setup

packages = find_packages(exclude=('tests', 'doc'))

provides = ['exogan', ]


requires = []


install_requires = ['numpy',
                    'configobj',
                    'scipy',
                    'matplotlib',
                    'numpy',
                    'h5py',
                    'tensorflow==1.15',
                    'tqdm',
                    'imageio',
                    'pandas',
                    'astropy']

console_scripts = ['exogan=exogan.exogan:main']

entry_points = {'console_scripts': console_scripts, }

classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Environment :: Win32 (MS Windows)',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Unix',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]

# Handle versioning
version = '2.0.0-alpha'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='exogan',
    author='Tiziano Zingales',
    author_email='tiziano.zingales@u-bordeaux.fr',
    license="LAB",
    version=version,
    description='ExoGAN 2 atmospheric analysis framework',
    classifiers=classifiers,
    packages=packages,
    long_description=long_description,
    url='https://github.com/ucl-exoplanets/ExoGAN2.git',
    long_description_content_type="text/markdown",
    keywords = ['exoplanet','retrieval','exogan', 'exogan2','atmosphere','atmospheric'],
    package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.txt", "*.rst", "*.dat"]},
    # include_package_data=True,
    entry_points=entry_points,
    provides=provides,
    requires=requires,
    install_requires=install_requires)
