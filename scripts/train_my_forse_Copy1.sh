#!/bin/bash
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -G 1
#SBATCH -A mp107
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

module load tensorflow/2.6.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/u2/j/jianyao/my_software/cfitsio/lib:/global/u2/j/jianyao/my_software/gsl/lib:/global/homes/j/jianyao/my_software/fftw/lib

cd /global/homes/j/jianyao/ForSEplus_github/scripts/
python train_my_forse_copy1.py --output_dirs ${output_dirs} --patch_file ${patch_file}


