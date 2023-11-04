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

First thing to do is to install the dependencies. 

- pymaster
The installation of pymaster is acutally not a easy task since it has a lot of other depencencies. 
- tensorflow

### from source
    pip install -e .

### from pip
    pip install forseplus
    
