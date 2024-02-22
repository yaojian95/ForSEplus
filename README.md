# ForSEplus
- plus version of https://github.com/ai4cmb/ForSE.

- If you want to use the generated maps directly, we also offer 500 realizations of maps at url...

# Installations
## Dependencies  

- Astropy: https://www.astropy.org
- Healpy: https://healpy.readthedocs.io/en/latest/
- Tensorflow: https://www.tensorflow.org
- Namaster (to compute power spectra to normalize the small scales from neural networks): https://namaster.readthedocs.io/en/latest/
- reproject (only to perform projection from Healpix maps to flat patches and viceversa): https://pypi.org/project/reproject/
- numba (only to accelearte the calculation of Minkowski functionals for a given patch): http://numba.pydata.org/

We assume you alrealy have your own python virtual environment. 
The first thing to do is to install the dependencies and the main difficulty is to install the `Namaster` package. If you are a NERSC user, we prepared a `install_dependencies.sh` file for you. 

Then you have two ways to install this package. 

## from source
Download the source code, then 

    (venv) $ python -m pip install -e . --user

## from pip (not updated with the latest source code yet)
    (venv) $ python -m pip install ForSEplus --user

# Ancillary data 
The zipped complete ancillary data can be downloaded at ... (13GB in total after decompression). Then decompress the files into a directory, whose path should be given to `dir_data` when running the pipeline. 

# Usage
Once installed, import the `forseplus` as:

    from ForSEplus.forseplus_class import forseplus
    
Then intialize a instance to generate maps:

    fp = forseplus(dir_data = '/pscratch/sd/j/jianyao/ForSE_plus_data/', 
            return_12 = True,
            go_3 = True,
            correct_EB = False)

and run:

    maps_12amin, maps_3amin = fp.run()
    
You can choose to return sotchastic maps at 12 arcmin only (`return_12 = True, go_3 = False`), or maps at 3 arcmin only (`return_12 = False, go_3 = True`), or both (`return_12 = True, go_3 = True`), though in any case maps at 12 arcmin will be generated since 12arcmin maps will be the input to generate maps at 3arcmin. 

If set `correct_EB = True`, it will apply the E/B ratio correction proposed in Yao et al. to artificially tune the Cl_EE/Cl_BB = 2 for the generated small scales. Otherwise, Cl_EE/Cl_BB = 1. Refer to Section 4.1 of Yao et al. for more details. 

If you want to generate many realizations, just put the `fp.run()` inside a loop. 

## Memory needed (Peak memory) and time cost (test on Perlmutter Jupyter Exclusive GPU node)
- only 12amin: CPU 16GB, GPU 10G; ~15 secs
- also go to 3amin: CPU 66GB*, GPU 18G; ~5 mins

* (On NERSC Perlmutter, this will exceed the memory limit (64GB) on login node, so you may open a Exclusive GPU node or submit the job to a compute node.)
    
