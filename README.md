# sidekit
This is a fork of the SIDEKIT python package (http://www-lium.univ-lemans.fr/sidekit/), forked from https://git-lium.univ-lemans.fr/Larcher/sidekit. 
It includes some custom modifications as well as documentation on how to work around some installation issues (now that theano is no longer maintained, `pip install sidekit` results in errors about pygpu).

## Linux

### Install Miniconda3:

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    md5sum Miniconda3-latest-Linux-x86_64.sh
    # For version 4.3.31 this should return: 7fe70b214bee1143e3e3f0467b71453c
    # For other versions, see https://repo.continuum.io/miniconda/
    bash Miniconda3-latest-Linux-x86_64.sh

### Create new virtual environment

    conda create --name my_env python=3

### Activate environment

    source activate my_env

### Install dependencies (Theano and working libsvm)

    conda install -c mila-udem -c mila-udem/label/pre theano pygpu
    conda install -c conda-forge libsvm=3.21
	
### Install sidekit

    pip install sidekit

### Set environment variables (add them to ~/.bashrc or something)

    MKL_THREADING_LAYER=GNU
