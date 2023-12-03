TODO 
====
- change path to some environment variable like os.environ['Forse_plus_data']
- change verbose to log
- add recomposition to full sky for 12amin
- treate Q and U separately!!! will save up to half of the memory used
- change float6 to float32!!
- correct_EB also consumes a lo of memory


2023-12-3 Peak RAM: 64GB
==========================
- set variables to be local ones;
- the post-training class can be created without the NN_out, which can be passed to the class during each iteration to generate many realizations;
- utility functions ``from_12to20`` and ``from_12to13`` are merged to be ``from_12toXX`` with XX = 12 or XX = 13;
- comment out correct_EB for now.

2023-11-4 1.0
================
- upload to pip (https://pypi.org/project/ForSEplus/).