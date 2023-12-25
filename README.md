# ForSEplus
- plus version of https://github.com/ai4cmb/ForSE.

- To avoid the installation this package (which is a bit complex), we also offer 500 realizations of maps at url...

## Dependencies  

- Astropy: https://www.astropy.org
- Healpy: https://healpy.readthedocs.io/en/latest/
- Tensorflow: https://www.tensorflow.org
- Namaster (to compute power spectra to normalize the small scales from neural networks): https://namaster.readthedocs.io/en/latest/
- reproject (only to perform projection from Healpix maps to flat patches and viceversa): https://pypi.org/project/reproject/
- numba (only to accelearte the calculation of Minkowski functionals for a given patch): http://numba.pydata.org/

## Installations
We assume you alrealy your own python virtual environment. 
The first thing to do is to install the dependencies and the main difficulty is to install the $Namaster$ package. If you are a NERSC user, we prepared a `install_dependencies.sh` file for you. 

Then you have two ways to install this package. 

### from source
    (venv) $ python -m pip install -e . --user

### from pip
    (venv) $ python -m pip install ForSEplus --user
    
