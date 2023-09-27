import numpy as np
import pymaster as nmt
import os 
import time 
from collections import Counter
from ForSEplus import after_training_3amin as at3
from ForSEplus import after_training_12amin as at12
from ForSEplus import utility, validation_class, check_model
import matplotlib.pyplot as plt

dir_data = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'
# ss_I = np.load(dir_data+'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[0, 0:174]

ss_I = np.load(dir_data+'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[0, 0:348]

# Ls_Q80amin = np.load(dir_data + 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6
# Ls_U80amin = np.load(dir_data + 'GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]*1e6

gauss_ss_ps_12 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_12_over_80_power_spectra.npy') #[2, 174, 49, 1, 25] Q, U
gauss_ss_mean_std_12 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_12_over_80_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

gauss_ss_ps_3 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_3_over_20_power_spectra_lmax_3500.npy') #[2, 174, 49, 1, 25] Q, U
gauss_ss_mean_std_3 = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_3_over_20_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std



def most_frequent(arr):
    freq_count = Counter(arr)
    return freq_count.most_common(1)[0][0]

def gen_test_MF(MF_dir, file_name, n_patches):
    '''
    generate MFs for testing dat after training 
    '''
    
    for n in range(2, 20, 8):
        stokes = ['Q', 'U']
        for s in stokes:
            # data_train = np.load('/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/GNILC_Thr12_%slr80_20x20deg_Npix320_full_sky_adaptive.npy'%s)
            MFs = MF_dir + 'Test_Random_random_3_arcmin_%sMY_%s_jupyter_with_seed_%03d_corrected_snr_1.txt'%(file_name, s, n);
            print(MFs)

            model_dir = '/pscratch/sd/j/jianyao/forse_output/Random_3_arcmin_%s_models_MY_lr_5e-5_%s_jupyter_revised_sto_12_rescaled_snr/'%(file_name, s)
            
            dir_name = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/Random_%s_snr_1_3amin/'%file_name
            test_data = np.load(dir_name + 'testing_data_Nico_T12amin_%s20amin_%s_random_snr_1_%03d.npy'%(s,file_name, n)) 

            if n == 1:
                plot_f = True
            else: 

                plot_f = False

            results = check_model.get_MFs_fix(input_patches=n_patches, data_dir = test_data, 
                                              model_dir = model_dir, MF_dir = MFs, checkpoint=test_forse.checkpoint, plot = plot_f)
            
def rescale_input(Ls, random_noise = None):
    Ls_rescaled = np.zeros_like(Ls)
    
    if random_noise is not None:
        assert random_noise.shape == Ls.shape
        
        for i in range(Ls.shape[0]):
            Ls_rescaled[i] = utility.rescale_min_max(utility.rescale_min_max(Ls[i]) + random_noise[i])
            
    else:   
        for i in range(Ls.shape[0]):
                Ls_rescaled[i] = utility.rescale_min_max(Ls[i])
                
    Ls_rescaled = Ls_rescaled.reshape((Ls.shape[0], Ls.shape[1], Ls.shape[1], 1)) 
    return Ls_rescaled

def pick_be(MF_dir, file_name = '245'):
    best_e_Q = []
    best_e_U = []
    best_e = []
    for n in range(2, 20, 8):
        stokes = ['Q', 'U']
        for s in stokes:

            MFs = MF_dir + 'Test_Random_random_3_arcmin_%s_MY_%s_jupyter_with_seed_%03d_corrected_snr_1.txt'%(file_name, s, n);
            print(s, n)

            data = np.loadtxt(MFs)
            mf_mean = np.mean(data[:, 1:4], axis = 1)
            index = np.argsort(mf_mean)[::-1]+1

            if s == 'Q':
                best_e_Q.append(index[0])
            else:
                best_e_U.append(index[0])
            print(index[:4], mf_mean[index[:4] - 1])
    best_e = [most_frequent(best_e_Q), most_frequent(best_e_U)]
    print('best_e:', best_e)
    
    return best_e

def flat_ps_new(maps, lmin, lmax, bw, mask_path, w22_file, side_length = 20):
    mask = np.load(mask_path)
    
    l0_bins = np.arange(lmin, lmax, bw); lf_bins = np.arange(lmin, lmax, bw)+(bw-1)
    # print(l0_bins)
    # print(lf_bins)
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()
    w22 = nmt.NmtWorkspaceFlat()
    
    Lx = np.radians(side_length); Ly = np.radians(side_length)   
    f_NN = nmt.NmtFieldFlat(Lx, Ly, mask, [maps[0], maps[1]], purify_b=False)
    
    try:
        w22.read_from(w22_file)
        
    except:
        w22.compute_coupling_matrix(f_NN, f_NN, b)
        w22.write_to(w22_file) 
        print('weights writing to disk')

    cl_NN_coupled = nmt.compute_coupled_cell_flat(f_NN, f_NN, b)
    cl_NN_uncoupled = w22.decouple_cell(cl_NN_coupled)

    return ells_uncoupled, cl_NN_uncoupled

def get_cov(cls_all, lmin, lmax, bw, log = False):
    
    '''
    cls_all: list, including the power spectra from namaster of all the realizations; in the shape of (N_sam, 4, N_ell)
    '''
    N_sam = len(cls_all[0])
    N_ell = (lmax - lmin) //bw #len(cls_all[0][0][0])
    
    print(N_sam, N_ell)
    cls_EE = np.zeros((N_sam, N_ell))
    cls_BB = np.zeros((N_sam, N_ell))
    
    for i in range(N_sam):
        cls_EE[i] = cls_all[0][i][0][0:N_ell]
        cls_BB[i] = cls_all[0][i][3][0:N_ell]

    corr_EE = np.corrcoef(cls_EE.T)
    corr_BB = np.corrcoef(cls_BB.T)
    
    if log == True:
        corr_EE = np.log10(corr_EE)
        corr_BB = np.log10(corr_BB)

    fig, axes = plt.subplots(1, 2, figsize = (10, 8))
    axes[0].imshow(corr_EE)
    axes[1].imshow(corr_BB)

    axes[0].set_xticks((0, 10, 15))
    axes[0].set_yticks((0, 10, 15))

    axes[0].set_xticklabels((0, 10*bw, 15*bw), fontsize = 12)   
    axes[0].set_yticklabels((0, 10*bw, 15*bw), fontsize = 12)   

    axes[1].set_xticks((0, 10, 15))
    axes[1].set_yticks((0, 10, 15))
    axes[1].set_xticklabels((0, 10*bw, 15*bw), fontsize = 12)   
    axes[1].set_yticklabels((None, None, None), fontsize = 12)  

    axes[0].set_title('EE')
    axes[1].set_title('BB')

def get_cls_patches(test_forse, model_dirs, best_e, pure_noise = False, reso = '12', j_range = np.arange(3), i_range = np.arange(100), lmin = 20, lmax = 1000, bw = 40, only_one = True, patch_id = 33, snr = 1):
    
    s = time.time()
    cls_NN = [];
    if only_one:
        N = 0
    else:
        N = 81
    if reso == '3':
        # lmax = 3000
        w22_file = "w22_flat_1280_1280_lmin_%s_lmax_%s_bw_%s_no_purifyB.fits"%(lmin, lmax, bw)
        mask_path = './src/ForSEplus/mask_1280x1280.npy'
    
        for j in j_range:
            cls_NN_j = []
            
            for i in i_range:
                # print(patch_id)
                data_Q, data_U = get_3amin_20by20(test_forse = test_forse, model_dirs = model_dirs, pure_noise = pure_noise, index = i, best_e = best_e, only_one = only_one, patch_id = patch_id, snr = snr)

                ells_uncoupled, cls_NN_i = flat_ps_new(maps = [data_Q[N], data_U[N]], lmin=lmin, lmax = lmax, bw = bw, w22_file = w22_file, mask_path = mask_path)
                cls_NN_j.append(cls_NN_i)
                
                if i != 0 and i%100 == 0:
                    e = time.time()
                    print('Averate time cost for each realization is %s mins'%((e-s)/60/(i+1)))

            cls_NN.append(cls_NN_j)
    return cls_NN

def get_3amin_20by20(test_forse, model_dirs, index, best_e, pure_noise = False, model = 3, only_ss = False, only_one = True, patch_id = 33, validate = False, test = False, snr = 1):
    '''
    snr: level of noise added to the input patches
    '''
    if only_one:
        
        assert patch_id is not False, 'a and b should be both True or both False'
        N_patch = 1
        patch_N = range(patch_id, patch_id + 1)
    
    else:
        assert patch_id is False, 'patch_id will be False when only_one is False, but it is %s'%patch_id
        N_patch = 174   
        patch_N = range(174)
        
    dir_name_12 = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/FIX_MF/2_random_12amin_renormalized/New_realizations_1/' 
    NN_12amin_Q = np.load(dir_name_12 + 'Random_1_testing_data_Nico_T12amin_1_Q80amin_renormalized_%03d.npy'%index)
    NN_12amin_U = np.load(dir_name_12 + 'Random_1_testing_data_Nico_T12amin_1_U80amin_renormalized_%03d.npy'%index)

    Ls_13aminQ, Ls_13aminU = utility.from_12to13(NN_12amin_Q[patch_N], NN_12amin_U[patch_N], only_one = only_one) # to normalize the output from 3amin

    if pure_noise is not False:
        Ls_rescaled_Q, Ls_rescaled_U = np.random.uniform(-1, 1, (N_patch*49, 320, 320, 1))/snr, np.random.uniform(-1, 1, (N_patch*49, 320, 320, 1))/snr
        
    elif pure_noise is False:
        # noise_1 = np.random.uniform(-1, 1, (174*49, 320, 320))
        noise_1 = np.random.uniform(-1, 1, (N_patch*49, 320, 320))/snr
        noise_2 = np.random.uniform(-1, 1, (N_patch*49, 320, 320))/snr

        Ls_20aminQ, Ls_20aminU = utility.from_12to20(NN_12amin_Q[patch_N], NN_12amin_U[patch_N], random_noise=None, only_one = only_one)     

        Ls_rescaled_Q, Ls_rescaled_U = rescale_input(Ls_20aminQ, random_noise=noise_1), rescale_input(Ls_20aminU, random_noise=noise_2)
    # print(Ls_rescaled_Q.shape)
    # del Ls_20aminQ, Ls_20aminU, noise_1
    
    if model == 3:
        test_forse.checkpoint.restore(model_dirs[0] + 'training_checkpoints/ckpt-%s'%best_e[0])
        NNout_Q = test_forse.checkpoint.generator.predict(Ls_rescaled_Q)
        # print(NNout_Q.shape)
        test_forse.checkpoint.restore(model_dirs[1] + 'training_checkpoints/ckpt-%s'%best_e[1])
        NNout_U = test_forse.checkpoint.generator.predict(Ls_rescaled_U)
        
#     elif model == 12:
#         test_forse.checkpoint.restore(model_dir_Q_12 + 'training_checkpoints/ckpt-%s'%11)
#         NNout_Q = test_forse.checkpoint.generator.predict(Ls_rescaled_Q)
#         test_forse.checkpoint.restore(model_dir_U_12 + 'training_checkpoints/ckpt-%s'%12)
#         NNout_U = test_forse.checkpoint.generator.predict(Ls_rescaled_U)
    
    if only_ss:
        
        return NNout_Q, NNout_U
    
    if validate:
        assert patch_id is False
        output3 = at3.post_training(NNout_Q, NNout_U, ss_I, Ls_13aminQ, Ls_13aminU, MF = True, patch_id = patch_id)
    else:
        output3 = at3.post_training(NNout_Q, NNout_U, ss_I, Ls_13aminQ, Ls_13aminU, MF = False, patch_id = patch_id)
    
    # del Ls_13aminQ, Ls_13aminU, NNout_Q, NNout_U
    
    output3.normalization(gauss_ss_ps_3, gauss_ss_mean_std_3, mask_path = './src/ForSEplus/mask_320x320.npy')
    
    if test:
        assert patch_id is not False
        return output3.NNmapQ_corr, output3.NNmapU_corr
    
    file_name_Q = 'NN_out_Q_3amin_from_20amin_physical_units_20x20_1280_%03d.npy'%(index)
    file_name_U = 'NN_out_U_3amin_from_20amin_physical_units_20x20_1280_%03d.npy'%(index)
    
    if only_one:
        output3.combine_to_20by20(output3.NNmapQ_corr, output3.NNmapU_corr, maps = 'ss_norm')
        
    else:
        output3.combine_to_20by20(output3.NNmapQ_corr, output3.NNmapU_corr, maps = 'ss_norm',save_dir=[save_dir + file_name_Q, save_dir + file_name_U])
    
    
    if validate:
        
        return output3
    
    else:
        
        return output3.NN_20by20_Q_norm, output3.NN_20by20_U_norm
    
def get_3amin_5by5(test_forse, model_dirs, index, best_e, pure_noise = False, model = 3, only_one = True, patch_id = 33, validate = False, test = False, snr = 1):
    '''
    snr: level of noise added to the input patches
    '''
    if only_one:
        
        assert patch_id is not False, 'a and b should be both True or both False'
        N_patch = 1
        patch_N = range(patch_id, patch_id + 1)
    
    else:
        assert patch_id is False, 'patch_id will be False when only_one is False, but it is %s'%patch_id
        N_patch = 174   
        patch_N = range(174)
        
    dir_name_12 = '/pscratch/sd/j/jianyao/forse_output/Random_training_files/FIX_MF/2_random_12amin_renormalized/New_realizations_1/' 
    NN_12amin_Q = np.load(dir_name_12 + 'Random_1_testing_data_Nico_T12amin_1_Q80amin_renormalized_%03d.npy'%index)
    NN_12amin_U = np.load(dir_name_12 + 'Random_1_testing_data_Nico_T12amin_1_U80amin_renormalized_%03d.npy'%index)

    Ls_13aminQ, Ls_13aminU = utility.from_12to13(NN_12amin_Q[patch_N], NN_12amin_U[patch_N], only_one = only_one) # to normalize the output from 3amin

    if pure_noise is not False:
        Ls_rescaled_Q, Ls_rescaled_U = np.random.uniform(-1, 1, (N_patch*49, 320, 320, 1))/snr, np.random.uniform(-1, 1, (N_patch*49, 320, 320, 1))/snr
        
    elif pure_noise is False:
        # noise_1 = np.random.uniform(-1, 1, (174*49, 320, 320))
        noise_1 = np.random.uniform(-1, 1, (N_patch*49, 320, 320))/snr
        noise_2 = np.random.uniform(-1, 1, (N_patch*49, 320, 320))/snr

        Ls_20aminQ, Ls_20aminU = utility.from_12to20(NN_12amin_Q[patch_N], NN_12amin_U[patch_N], random_noise=None, only_one = only_one)     

        Ls_rescaled_Q, Ls_rescaled_U = rescale_input(Ls_20aminQ, random_noise=noise_1), rescale_input(Ls_20aminU, random_noise=noise_2)
    # print(Ls_rescaled_Q.shape)
    # del Ls_20aminQ, Ls_20aminU, noise_1
    
    if model == 3:
        test_forse.checkpoint.restore(model_dirs[0] + 'training_checkpoints/ckpt-%s'%best_e[0])
        NNout_Q = test_forse.checkpoint.generator.predict(Ls_rescaled_Q)
        # print(NNout_Q.shape)
        test_forse.checkpoint.restore(model_dirs[1] + 'training_checkpoints/ckpt-%s'%best_e[1])
        NNout_U = test_forse.checkpoint.generator.predict(Ls_rescaled_U)
    
    if validate:
        assert patch_id is False
        output3 = at3.post_training(NNout_Q, NNout_U, ss_I, Ls_13aminQ, Ls_13aminU, MF = True, patch_id = patch_id)
    else:
        output3 = at3.post_training(NNout_Q, NNout_U, ss_I, Ls_13aminQ, Ls_13aminU, MF = False, patch_id = patch_id)
    
    output3.normalization(gauss_ss_ps_3, gauss_ss_mean_std_3, mask_path = './src/ForSEplus/mask_320x320.npy')
    
    if test:
        assert patch_id is not False
        return NNout_Q, NNout_U, output3.NNmapQ_corr, output3.NNmapU_corr   
    
def flat_ps_5by5(test_forse, model_dirs, best_e, j_range, pure_noise = False, NN = True, collect = True, N = 100, patch_id = 81, snr = 1):
    lmax = 4000
    # w22_file = "w22_flat_320_320_lmax_1000_no_purifyB.fits"
    w22_file = "w22_flat_320_320_lmin_80_lmax_4000_no_purifyB_bw_160.fits"
    mask_path = './src/ForSEplus/mask_320x320.npy'

    j_range = j_range #np.arange(20, 21) #### test for sub-patch 20
    cls_NN_all = np.zeros((N, len(j_range), 4, 25))
    
    if collect == True:
        Qall = np.zeros((N, 49, 320, 320)); Uall = np.zeros((N, 49, 320, 320))
        
    for i in range(N):
        if NN:
            NN_norm_3Q, NN_norm_3U = get_3amin_20by20(test_forse, model_dirs, pure_noise = pure_noise, index = 10, best_e = best_e, only_one = True, patch_id = patch_id, validate = False, test = True, model=3, snr = snr)
        else:
            NN_norm_3Q, NN_norm_3U = np.random.random((1, 49, 320, 320)), np.random.random((1, 49, 320, 320))

        # if i % 20 == 0:
            # print(i)
        k = 0
        for j in j_range:

            ells_uncoupled, cls_NN_j = flat_ps_new(maps = [NN_norm_3Q[0][j], NN_norm_3U[0][j]], lmin = 80, lmax = lmax, w22_file = w22_file, mask_path = mask_path, bw = 160, side_length=5)
            cls_NN_all[i, k] = np.copy(cls_NN_j)

            k += 1
            
        if collect == True:
            Qall[i] = NN_norm_3Q[0]; Uall[i] = NN_norm_3U[0]
            
    cls_all = np.zeros((len(j_range), N, 4, 25))
    for i in range(len(j_range)):

        cls_all[i] = cls_NN_all[:, i, :, :]
        

    if collect == True:
        return cls_all, [Qall, Uall]
    else:
        return cls_all