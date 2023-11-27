import os

# os.environ[
#     "OMP_NUM_THREADS"
# ] = "64"  # for jupyter.nersc.gov otherwise the notebook only uses 2 cores

# import matplotlib.pyplot as plt
# %matplotlib inline

# import matplotlib as mpl
# mpl.rc('image', cmap='coolwarm')

# import seaborn as sns

# sns.set_context("talk")
# # sns.set()
# sns.set_style("ticks")

import numpy as np
import healpy as hp

import logging
log = logging.getLogger("healpy")
log.setLevel(logging.ERROR)

import time 

from ForSEplus.my_forse_class import forse_my
from ForSEplus.utility import rescale_input, correct_EB, from_12to13, from_12to20
from ForSEplus.recompose_class import recom
from ForSEplus.after_training_12amin import post_training as post_training_12
from ForSEplus.after_training_3amin import post_training as post_training_3

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for g in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[g], True)

dir_data = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'
# ss_I = np.load(dir_data+'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[0, 0:174]

ss_I = np.load(dir_data+'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[0, 0:348]

Ls_Q80amin = np.load(dir_data + 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6
Ls_U80amin = np.load(dir_data + 'GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6

gauss_ss_ps_12 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_12_over_80_power_spectra.npy') #[2, 174, 49, 1, 25] Q, U
gauss_ss_mean_std_12 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_12_over_80_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

gauss_ss_ps_3 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_3_over_20_power_spectra_lmax_3500.npy') #[2, 174, 49, 1, 25] Q, U
gauss_ss_mean_std_3 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_3_over_20_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

ori_train_Q = np.load('/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy')[1, 0:174]
ori_train_U = np.load('/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy')[1, 0:174]

model_Q_12amin = '/pscratch/sd/j/jianyao/forse_output/Random_model_PySM/model_all_h5_snr_1_Q11'
model_U_12amin = '/pscratch/sd/j/jianyao/forse_output/Random_model_PySM/model_all_h5_snr_1_U12'

model_Q_3amin =  '/pscratch/sd/j/jianyao/forse_output/Random_model_PySM/model_3amin_696_patches_Q85'
model_U_3amin =  '/pscratch/sd/j/jianyao/forse_output/Random_model_PySM/model_3amin_696_patches_U396'

model_Q_det_3amin = '/pscratch/sd/j/jianyao/forse_output/Random_model_PySM/model_det_3amin_Q154'
model_U_det_3amin = '/pscratch/sd/j/jianyao/forse_output/Random_model_PySM/model_det_3amin_U256'

start = time.time()
print('Loading models')
model_Q = tf.keras.models.load_model(model_Q_12amin)
model_U = tf.keras.models.load_model(model_U_12amin)

model_Q_3 = tf.keras.models.load_model(model_Q_3amin)
model_U_3 = tf.keras.models.load_model(model_U_3amin)

print('Initialize the recomposing class')
recom_3 = recom(npix = 1280, pixelsize = 0.9375, overlap = 2, nside = 4096, 
                 apodization_file = '/pscratch/sd/j/jianyao/mask_1280*1280.npy', 
                 xy_inds_file = '/pscratch/sd/j/jianyao/forse_recompose/recompose_xinds_yinds_4096', 
                 index_sphere_file = '/pscratch/sd/j/jianyao/forse_recompose/recompose_footprint_healpix_index_4096', verbose=False)

print('Load maps at 80amin')
Ls_Q = ori_train_Q.copy()
Ls_U = ori_train_U.copy()

print('Generating input random noise with model SNR = %s...'%1)
# np.random.seed(2048)


verbose = False

for i in range(1, 100):
    noise_1_12 = np.random.uniform(-1, 1, (174, 320, 320))
    noise_2_12 = np.random.uniform(-1, 1, (174, 320, 320))

    noise_1 = np.random.uniform(-1, 1, (174*49, 320, 320))
    noise_2 = np.random.uniform(-1, 1, (174*49, 320, 320))

    ratio = 1
    Ls_rescaled_Q = rescale_input(Ls_Q, random_noise = noise_1_12)
    Ls_rescaled_U = rescale_input(Ls_U, random_noise = noise_2_12)
    
    if verbose == True:
        print('12amin Generating patches...')

    NNout_Q_12 = model_Q.predict(Ls_rescaled_Q)
    NNout_U_12 = model_U.predict(Ls_rescaled_U)

    if verbose == True:
        print('12amin: Renormalize patches...')
    output12 = post_training_12(NNout_Q_12[:,:,:,0] , NNout_U_12[:,:,:,0], ss_I, Ls_Q80amin, Ls_U80amin, MF = False)
    save_dir_12 = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/FIX_MF/Complete/12amin_norm/'
    norm_Q12amin = 'NN_out_Q_12amin_from_12amin_physical_units_20x20_320_%03d.npy'%(i)
    norm_U12amin = 'NN_out_U_12amin_from_12amin_physical_units_20x20_320_%03d.npy'%(i)
        
    output12.normalization(gauss_ss_ps_12, gauss_ss_mean_std_12, mask_path = '/global/homes/j/jianyao/ForSEplus_github/src/ForSEplus/mask_320x320.npy', save_path = [save_dir_12 + norm_Q12amin, save_dir_12 + norm_U12amin])

    # output12.plot_MF()
        # dir_12 = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/FIX_MF/2_random_12amin_renormalized/New_realizations_1/'
        # NNmapQ_corr = np.load(dir_12 + 'Random_1_testing_data_Nico_T12amin_1_Q80amin_renormalized_000.npy')
        # NNmapU_corr = np.load(dir_12 + 'Random_1_testing_data_Nico_T12amin_1_U80amin_renormalized_000.npy')
    
    if verbose == True:
        print('3 amin: construct input 20amin and normalizing-13amin maps')
    Ls_13aminQ, Ls_13aminU = from_12to13(output12.NNmapQ_corr, output12.NNmapU_corr) # to normalize the output from 3amin

    Ls_20aminQ, Ls_20aminU = from_12to20(output12.NNmapQ_corr, output12.NNmapU_corr) # to be the input for the 3amin
    del output12

    Ls_rescaled_Q, Ls_rescaled_U = rescale_input(Ls_20aminQ, random_noise=noise_1), rescale_input(Ls_20aminU, random_noise=noise_2)

    if verbose == True:
        print('3amin: Generating patches...')

    NNout_Q = model_Q_3.predict(Ls_rescaled_Q)
    NNout_U = model_U_3.predict(Ls_rescaled_U)

    if verbose == True:
        print('3amin: renormalize patches ...')
    output3 = post_training_3(NNout_Q, NNout_U, ss_I, Ls_13aminQ, Ls_13aminU, MF = False)
    output3.normalization(gauss_ss_ps_3, gauss_ss_mean_std_3, mask_path = '/global/homes/j/jianyao/ForSEplus_github/src/ForSEplus/mask_320x320.npy')
    norm_Q3amin = 'NN_out_Q_3amin_from_20amin_physical_units_20x20_1280_%03d.npy'%(i)
    norm_U3amin = 'NN_out_U_3amin_from_20amin_physical_units_20x20_1280_%03d.npy'%(i)
    save_dir_3 = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/FIX_MF/Complete/3amin_norm/'
    output3.combine_to_20by20(output3.NNmapQ_corr, output3.NNmapU_corr, maps = 'ss_norm', save_path=[save_dir_3 + norm_Q3amin, save_dir_3 + norm_U3amin])
    # output3.combine_to_20by20(NNout_Q.reshape(174,49,320,320), NNout_U.reshape(174,49,320,320), maps = 'ss')
    # test = output3.plot_MF(patch_N = 3, savedir=False)

#     if verbose == True:
#         print('3amin: reproject to full sky ...')
#     full_Q = recom_3.recompose_fast(output3.NN_20by20_Q_norm)
#     full_U = recom_3.recompose_fast(output3.NN_20by20_U_norm) 

#     maps_3amin = correct_EB(full_Q, full_U)

#     del output3
#     end = time.time()

#     save_full_dir = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/FIX_MF/5_New_3amin_model_full_sky_EB_fixed/'
#     save_full_Q = save_full_dir + 'Random_3amin_full_Q_%03d.fits'%(i)
#     save_full_U = save_full_dir + 'Random_3amin_full_U_%03d.fits'%(i)

#     hp.write_map(save_full_Q, maps_3amin[0], overwrite=True)
#     hp.write_map(save_full_U, maps_3amin[1], overwrite=True)

print('Time cost %.02f'%((end - start)/60))