import numpy as np

from ForSEplus.my_forse_class import forse_my
from ForSEplus.utility import rescale_input, correct_EB, from_12toXX
from ForSEplus.recompose_class import recom
from ForSEplus.after_training_12amin import post_training as post_training_12
from ForSEplus.after_training_3amin import post_training as post_training_3

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
for g in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[g], True)

class forseplus:
    
    def __init__(
        self, 
        dir_data = '/pscratch/sd/j/jianyao/ForSE_plus_data/', 
        return_12 = False,
        go_3 = False,
        correct_EB = False, 
        plot_MF = False):
        
        '''
        Parameters
        ----------

        dir_data: Str; Path of ForSE_plus_data which includes all the ancillary data;
        return_12: Bool; If True, full-sky maps at 12 arcmin will also be returned;
        go_3: Bool; If False, will only generate stochastic maps at 12 arcmin;
        correct_EB: Bool; If True, apply the E/B ratio correction proposed in Yao et al. 
        plot_MF: Bool; If True, will also plot the MF overlapping fractions between generated maps and the ground-truth maps, as shown in Yao et al.

        Return
        ------
        list of arrays.
        
        '''
    
        self.dir_data = dir_data 
        
        self.Ls_Q80amin = np.load(dir_data + 'maps_cls/GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6
        self.Ls_U80amin = np.load(dir_data + 'maps_cls/GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6

        self.gauss_ss_ps_12 = np.load(dir_data + 'maps_cls/gauss_small_scales_12_over_80_power_spectra.npy') #[2, 174, 49, 1, 25] Q, U
        self.gauss_ss_mean_std_12 = np.load(dir_data + 'maps_cls/gauss_small_scales_12_over_80_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

        model_Q_12amin = dir_data + 'models/model_all_h5_snr_1_Q11'
        model_U_12amin = dir_data + 'models/model_all_h5_snr_1_U12'

        ss_I = np.load(dir_data + 'maps_cls/GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[0, 0:348]
        self.output12 = post_training_12(ss_I, self.Ls_Q80amin, self.Ls_U80amin, MF = plot_MF)
        self.model_Q = tf.keras.models.load_model(model_Q_12amin, compile = False)
        self.model_U = tf.keras.models.load_model(model_U_12amin, compile = False)
        
        self.return_12 = return_12
        if return_12:
            self.recom_12 = recom(npix = 320, pixelsize = 3.75, overlap = 2, nside = 2048, 
                             apodization_file = dir_data + 'masks/mask_320x320.npy', 
                             xy_inds_file = dir_data + 'geometry/recompose_xinds_yinds_2048', 
                             index_sphere_file = dir_data + 'geometry/recompose_footprint_healpix_index_2048', verbose=False)    
        self.go_3 = go_3
        if go_3:

            self.gauss_ss_ps_3 = np.load(dir_data + 'maps_cls/gauss_small_scales_3_over_20_power_spectra_lmax_3500.npy') #[2, 174, 49, 1, 25] Q, U
            self.gauss_ss_mean_std_3 = np.load(dir_data + 'maps_cls/gauss_small_scales_3_over_20_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

            # Ls_Q = np.load(dir_data + 'maps_cls/GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy')[1, 0:174]
            # Ls_U = np.load(dir_data + 'maps_cls/GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy')[1, 0:174]

            # data 3.0G
            model_Q_3amin =  dir_data + 'models/model_3amin_696_patches_Q85'
            model_U_3amin =  dir_data + 'models/model_3amin_696_patches_U396'

            self.output3 = post_training_3(ss_I, MF = plot_MF)
            self.model_Q_3 = tf.keras.models.load_model(model_Q_3amin, compile = False)
            self.model_U_3 = tf.keras.models.load_model(model_U_3amin, compile = False)
        
        # model 1.0 G

            # recom class: 8G 
            self.recom_3 = recom(npix = 1280, pixelsize = 0.937, overlap = 2, nside = 4096, 
                             apodization_file = dir_data + 'masks/mask_1280x1280.npy', 
                             xy_inds_file = dir_data + 'geometry/recompose_xinds_yinds_4096', 
                             index_sphere_file = dir_data + 'geometry/recompose_footprint_healpix_index_4096', verbose=False)
            
        self.plot_MF = plot_MF
        self.correct_EB = correct_EB
    
    def run_12(self, return_12, plot_MF = False, correct_EB = False):
        
        self.output12.import_NNout(self.model_Q.predict(rescale_input([self.Ls_Q80amin], random_noise = [np.random.uniform(-1, 1, (174, 320, 320))])), stokes = 'Q')
        self.output12.import_NNout(self.model_U.predict(rescale_input([self.Ls_U80amin], random_noise = [np.random.uniform(-1, 1, (174, 320, 320))])), stokes = 'U')
        
        if plot_MF:
            test = self.output12.plot_MF()
            
        self.output12.normalization(self.gauss_ss_ps_12, self.gauss_ss_mean_std_12, mask_path = self.dir_data + 'masks/mask_320x320.npy')       

        if return_12:
            full_Q_12 = self.recom_12.recompose_fast(self.output12.NNmapQ_corr)
            full_U_12 = self.recom_12.recompose_fast(self.output12.NNmapU_corr) 
            maps_12amin = np.array((full_Q_12, full_U_12))
        
            if correct_EB:
                print('12amin: correct the E/B ratio')
                maps_12amin = correct_EB(full_Q_12, full_U_12, reso = '12amin')

            return maps_12amin
        
        return 0
    
    def run_3(self, plot_MF = False, correct_EB = False):
        
        # 6.5G # 6.5G
        self.output3.import_NNout(self.model_Q_3.predict(rescale_input([from_12toXX(self.output12.NNmapQ_corr, XX = 20)], random_noise = [np.random.uniform(-1, 1, (174*49, 320, 320))])), stokes = 'Q')
        self.output3.import_NNout(self.model_U_3.predict(rescale_input([from_12toXX(self.output12.NNmapU_corr, XX = 20)], random_noise = [np.random.uniform(-1, 1, (174*49, 320, 320))])), stokes = 'U')

        self.output3.normalization(self.gauss_ss_ps_3, self.gauss_ss_mean_std_3, self.output12.NNmapQ_corr, self.output12.NNmapU_corr, 
                              mask_path = self.dir_data + 'masks/mask_320x320.npy')

        self.output3.combine_to_20by20(self.output3.NNmapQ_corr, self.output3.NNmapU_corr, maps = 'ss_norm')
        
        if plot_MF:
            test = self.output3.plot_MF(patch_N = 3, savedir=False)

        full_Q = self.recom_3.recompose_fast(self.output3.NN_20by20_Q_norm)
        full_U = self.recom_3.recompose_fast(self.output3.NN_20by20_U_norm)
        maps_3amin = np.array((full_Q, full_U))
        
        if correct_EB:
            print('3amin: correct the E/B ratio')
            maps_3amin = correct_EB(full_Q, full_U, reso = '3amin') # takes about 4 mins.

        return maps_3amin
    
    def run(self):
        '''
        To generate realizations of maps at 12 arcmin and 3 arcmin. `run_12` is run first whose results are the input of `run_3`.
        '''

        maps_12amin = self.run_12(self.return_12, plot_MF = self.plot_MF, correct_EB = self.correct_EB)
        
        if self.go_3:
        
            maps_3amin = self.run_3(plot_MF = self.plot_MF, correct_EB = self.correct_EB)

            # hp.write_map(dir_data + '3amin_full/Random_3amin_full_Q_%03d.fits'%(i), maps_3amin[0], overwrite=True)
            # hp.write_map(dir_data + '3amin_full/Random_3amin_full_U_%03d.fits'%(i), maps_3amin[1], overwrite=True)
            
            if self.return_12:

                return maps_12amin, maps_3amin
        
            else:
                return maps_3amin
            
        return maps_12amin
    
    
