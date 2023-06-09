import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
import numpy as np
from .after_training_12amin import post_training as post_training_12
from .after_training_3amin import post_training as post_training3
from .utility import from_12to13, from_12to20, rescale_input, rescale_min_max, correct_EB
from .recompose_class import recom

class forsev2(object):
    
    '''
    Generate polarized thermal dust emission with random realizations of small scales.
    Small scales can reach resolution of 12 arcminutes and 3 arcminutes.   
    
    Parameters
    ----------
    '''
    
    def __init__(self, data_dir, snr = 1, go_three = True, three_only = True, correct_EB = True, validation = False):
        
        maps_dir = data_dir + 'ForSE_plus_data/maps/'
        models_dir = data_dir + 'ForSE_plus_data/models/'
        self.geometry_dir = data_dir + 'ForSE_plus_data/geometry/'
        self.mask_dir = data_dir + 'ForSE_plus_data/masks/'
        self.correct_EB = correct_EB
    
        self.go_three = go_three
        self.snr = snr
        
        if three_only:
            assert go_three == True
            
        self.three_only = three_only
        
        if validation:
            pass
        
        self.ss_I = np.load(maps_dir+'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[0, 0:348]

        self.Ls_Q80amin = np.load(maps_dir + 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6
        self.Ls_U80amin = np.load(maps_dir + 'GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6

        self.gauss_ss_ps_12 = np.load(maps_dir + 'gauss_small_scales_12_over_80_power_spectra.npy') #[2, 174, 49, 1, 25] Q, U
        self.gauss_ss_mean_std_12 = np.load(maps_dir + 'gauss_small_scales_12_over_80_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

        self.ori_train_Q = np.load(maps_dir + 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy')[1, 0:174]
        self.ori_train_U = np.load(maps_dir + 'GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy')[1, 0:174]
        
        # load model first, then can be used to generate realizations for many times
        if snr == 10:
            self.ratio = 10
            self.model_Q_12amin = tf.keras.models.load_model(models_dir + 'model_all_h5_snr_10_Q9')
            self.model_U_12amin = tf.keras.models.load_model(models_dir + 'model_all_h5_snr_10_U6')
        elif snr == 1:
            self.ratio = 1;
            self.model_Q_12amin = tf.keras.models.load_model(models_dir + 'model_all_h5_snr_1_Q11')
            self.model_U_12amin = tf.keras.models.load_model(models_dir + 'model_all_h5_snr_1_U12')
        elif snr == 0.1:
            self.ratio = 0.1
            self.model_Q_12amin = tf.keras.models.load_model(models_dir + 'model_all_h5_snr_0p1_Q22')
            self.model_U_12amin = tf.keras.models.load_model(models_dir + 'model_all_h5_snr_0p1_U296')
            
        if go_three:
            self.gauss_ss_ps_3 = np.load(maps_dir + 'gauss_small_scales_3_over_20_power_spectra_lmax_3500.npy') #[2, 174, 49, 1, 25] Q, U
            self.gauss_ss_mean_std_3 = np.load(maps_dir + 'gauss_small_scales_3_over_20_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std
            
            self.model_Q_3amin = tf.keras.models.load_model(models_dir + 'model_3amin_Q154')
            self.model_U_3amin = tf.keras.models.load_model(models_dir + 'model_3amin_U256')
            
            # model_Q_3 = tf.keras.models.load_model(self.model_Q_3amin)
            # model_U_3 = tf.keras.models.load_model(self.model_U_3amin)           
    
    def generate_maps_12(self):
        
        '''
    
        Parameters
        ----------
    
        Returns
        -------
        
        (when self.three_only is not True) maps_12amin: numpy.array.
            with shape (2, 12*2048**2). Healpix maps of QU at NSIDE = 2048;
        patches_12amin: 
            normalized square patches of QU before reprojecting to full sky.
        
        '''
        start = time.time()
        
        print('12amin: Start!')
        print('12amin: Generating input random noise with model SNR = %s...'%self.snr)
        # np.random.seed(2048)
        Ls_Q = self.ori_train_Q.copy()
        Ls_U = self.ori_train_U.copy()
        noise = np.random.uniform(-1, 1, (174, 320, 320))

        for i in range(174):
            Ls_Q[i] = rescale_min_max(Ls_Q[i]) + noise[i]/self.ratio
            Ls_U[i] = rescale_min_max(Ls_U[i]) + noise[i]/self.ratio

        Ls_rescaled_Q = rescale_input(Ls_Q)
        Ls_rescaled_U = rescale_input(Ls_U)

        print('12amin Generating patches...')
        # model_Q = tf.keras.models.load_model(self.model_Q_12amin)
        # model_U = tf.keras.models.load_model(self.model_Q_12amin)

        NNout_Q_12 = self.model_Q_12amin.predict(Ls_rescaled_Q)
        NNout_U_12 = self.model_U_12amin.predict(Ls_rescaled_U)

        print('12amin: Renormalize patches...')
        output12 = post_training_12(NNout_Q_12[:,:,:,0] , NNout_U_12[:,:,:,0], self.ss_I, self.Ls_Q80amin, self.Ls_U80amin, MF = True)
        output12.normalization(self.gauss_ss_ps_12, self.gauss_ss_mean_std_12, mask_path = self.mask_dir + 'mask_320*320.npy')
        output12.plot_MF()
        
        if self.three_only:
            
            print('12amin: Finishing without reprojecting to full-sky')
            return np.array((output12.NNmapQ_corr, output12.NNmapU_corr))
        
        recom_12 = recom(npix = 320, pixelsize = 3.75, overlap = 2, nside = 2048, 
                         apodization_file = self.mask_dir + 'mask_320*320.npy', 
                         xy_inds_file = self.geometry_dir + 'recompose_xinds_yinds_2048', 
                         index_sphere_file = self.geometry_dir + 'recompose_footprint_healpix_index_2048', verbose=True)

        print('12amin: reproject to full sky ...')
        full_Q_12 = recom_12.recompose_fast(output12.NNmapQ_corr)
        full_U_12 = recom_12.recompose_fast(output12.NNmapU_corr) 
        
        maps_12amin = np.array((full_Q_12, full_U_12))
        
        if self.correct_EB:
            print('12amin: correct the E/B ratio')
            maps_12amin = correct_EB(full_Q_12, full_U_12, reso = '12amin')
        
        print('12amin: Finishing after reprojecting to full-sky')
        end_12 = time.time()
        print('Time cost to 12amin is: %.02f minutes!'%((end_12 - start)/60))
        
        return maps_12amin, np.array((output12.NNmapQ_corr, output12.NNmapU_corr))
    
    def generate_maps_3(self, QU_patches_12):

        '''
    
        Parameters
        ----------

        QU_patches_12: 
            QU patches at 12amin
        
        Returns
        -------
        maps_3amin: numpy.array
            with shape (2, 12*4096**2). Healpix maps of QU at NSIDE = 4096
        
        '''
        
        print('3amin: Start!')
        Ls_13aminQ, Ls_13aminU = from_12to13(QU_patches_12[0], QU_patches_12[1]) # to normalize the output from 3amin

        Ls_20aminQ, Ls_20aminU = from_12to20(QU_patches_12[0], QU_patches_12[1]) # to be the input for the 3amin

        Ls_rescaled_Q, Ls_rescaled_U = rescale_input(Ls_20aminQ), rescale_input(Ls_20aminU)

        print('3amin: Generating patches...')

        NNout_Q = self.model_Q_3amin.predict(Ls_rescaled_Q)
        NNout_U = self.model_U_3amin.predict(Ls_rescaled_U)

        print('3amin: renormalize patches ...')
        output3 = post_training3(NNout_Q, NNout_U, self.ss_I, Ls_13aminQ, Ls_13aminU, MF = True)
        output3.normalization(self.gauss_ss_ps_3, self.gauss_ss_mean_std_3, mask_path = self.mask_dir + 'mask_320*320.npy')
        output3.combine_to_20by20(output3.NNmapQ_corr, output3.NNmapU_corr, maps = 'ss_norm')
        output3.combine_to_20by20(NNout_Q.reshape(174,49,320,320), NNout_U.reshape(174,49,320,320), maps = 'ss')
        test = output3.plot_MF(patch_N = 3, savedir=False)

        recom_3 = recom(npix = 1280, pixelsize = 0.9375, overlap = 2, nside = 4096, 
                         apodization_file = self.mask_dir + 'mask_1280*1280.npy', 
                         xy_inds_file = self.geometry_dir + 'recompose_xinds_yinds_4096', 
                         index_sphere_file = self.geometry_dir + 'recompose_footprint_healpix_index_4096', verbose=True)

        print('3amin: reproject to full sky ...')
        full_Q = recom_3.recompose_fast(output3.NN_20by20_Q_norm)
        full_U = recom_3.recompose_fast(output3.NN_20by20_U_norm) 
        maps_3amin = np.array((full_Q, full_U))

        if self.correct_EB:
            print('3amin: correct the E/B ratio')
            maps_3amin = correct_EB(full_Q, full_U, reso = '3amin') # takes about 4 mins.

        del output3
        
        return maps_3amin
    
    def run(self, nsim = 1):

        start = time.time()
        
        if not self.three_only:
            
            maps_12_all = np.zeros((nsim, 2, 12*2048**2))
        
        if self.go_three:
            
            maps_3_all = np.zeros((nsim, 2, 12*4096**2))
            
        for i in range(nsim):
            
            if self.three_only:
                
                QU_patches_12 = self.generate_maps_12()
                maps_3_all[i] = self.generate_maps_3(QU_patches_12)
            
            elif self.go_three:
                
                maps_12_all[i], QU_patches_12 = self.generate_maps_12()
                maps_3_all[i] = self.generate_maps_3(QU_patches_12)
            
            elif not self.go_three:
                
                maps_12_all[i], QU_patches_12 = self.generate_maps_12()

                
        end = time.time()
        print('Total time cost is: %.02f minutes!'%((end - start)/60))
        
        if self.three_only:
            
            return maps_3_all
        
        elif self.go_three:
            
            return maps_12_all, maps_3_all
        
        elif not self.go_three:
            
            return maps_12_all