# Assuming tensorflow is already installed in your environment:-).


####################### Install pymaster #######################

# downloading to /pscratch/sd/j/jianyao/software_download
# install cfitsio
./configure --prefix=/global/u2/j/jianyao/my_software/cfitsio
make
make install 

# install fftw
./configure --prefix=/global/u2/j/jianyao/my_software/fftw/ --enable-openmp --enable-shared
make 
make install 

module load cray-fftw/3.3.8.12

# export FFTW_DIR=/opt/cray/pe/fftw/3.3.8.12/x86_milan/
export FFTW_DIR=/global/homes/j/jianyao/my_software/fftw/
export CFITSIO_DIR=/global/u2/j/jianyao/my_software/cfitsio/
export GSL_DIR=/global/u2/j/jianyao/my_software/gsl/

export LDFLAGS+=" -L$GSL_DIR/lib -L$CFITSIO_DIR/lib -L/$FFTW_DIR/lib"
export CPPFLAGS+=" -I$GSL_DIR/include -I$CFITSIO_DIR/include -I$FFTW_DIR/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GSL_DIR/lib:$CFITSIO_DIR/lib:$FFTW_DIR/lib
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic

LDSHARED="gcc -shared" CC=gcc python -m pip install pymaster --user

pip install reproject --user
pip install healpy --user
pip install numba --user