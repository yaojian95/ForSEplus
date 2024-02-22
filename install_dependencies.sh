# Assuming tensorflow is already installed in your environment:-).
# Enter your Python environment

module load tensorflow

####################### Install pymaster #######################
## change /global/u2/j/jianyao/ and /pscratch/sd/j/jianyao/ to your own path. 

my_home="/global/u2/j/jianyao/my_software"
my_soft="/pscratch/sd/j/jianyao/software_download"

mkdir $my_home
mkdir $my_soft
cd $my_soft

# downloading to /pscratch/sd/j/jianyao/software_download
wget https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio_latest.tar.gz
wget https://www.fftw.org/fftw-3.3.10.tar.gz
wget https://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz

tar -xzf gsl-latest.tar.gz --transform 's:^[^/]*:gsl:'
tar -xzf cfitsio_latest.tar.gz --transform 's:^[^/]*:cfitsio:'
tar -xzf fftw-3.3.10.tar.gz --transform 's:^[^/]*:fftw:'

# install cfitsio
cd cfitsio
./configure --prefix=$my_home/cfitsio
make
make install 

# install fftw
# module load cray-fftw/3.3.8.12
cd ../fftw
./configure --prefix=$my_home/fftw/ --enable-openmp --enable-shared
make 
make install 

cd ../gsl
./configure --prefix=$my_home/gsl
make
make install 

# export FFTW_DIR=/opt/cray/pe/fftw/3.3.8.12/x86_milan/
export FFTW_DIR=$my_home/fftw/
export CFITSIO_DIR=$my_home/cfitsio/
export GSL_DIR=$my_home/gsl/

export LDFLAGS+=" -L$GSL_DIR/lib -L$CFITSIO_DIR/lib -L/$FFTW_DIR/lib"
export CPPFLAGS+=" -I$GSL_DIR/include -I$CFITSIO_DIR/include -I$FFTW_DIR/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GSL_DIR/lib:$CFITSIO_DIR/lib:$FFTW_DIR/lib
export CRAYPE_LINK_TYPE=dynamic
export XTPE_LINK_TYPE=dynamic

LDSHARED="gcc -shared" CC=gcc python -m pip install pymaster --user


####################### Install other dependencies #######################
pip install reproject --user
pip install healpy --user
pip install numba --user

## Below is option.
###################### generate a jupyter kernel  call my_forse, which has pymaster installed ###############################
python -m ipykernel install --user --name my_forse --display-name my_forse

cd $HOME/.local/share/jupyter/kernels/my_forse
file=kernel.json

sed -i '3d' $file
sed -i '3i \  \"{resource_dir}/kernel-helper.sh",' $file
sed -i '4i \  \"python",' $file

file="kernel-helper.sh"
echo "#!/bin/bash" >> $file
echo "module load tensorflow" >> $file
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$my_home//cfitsio/lib:$my_home/gsl/lib:$my_home/fftw/lib" >> $file
echo "exec \"\$@\"" >> $file

chmod u+x kernel-helper.sh