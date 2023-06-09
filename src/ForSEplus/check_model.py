from .utility import rescale_min_max, get_functionals_fix, compute_intersection
import numpy as np
import os
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='coolwarm')


def get_MFs_fix(input_patches = 348, data_dir = '/pscratch/sd/j/jianyao/forse_output/training_data_Nico_T12amin_U20amin_348.npy', \
            model_dir = '/pscratch/sd/j/jianyao/forse_output/3_arcmin_8000_models_MY_lr_5e-5_U/', MF_dir = "MFs/3_arcmin_MY_U_jupyter_348_MFs_lr_5e-5.txt", \
           best_epoch = False, checkpoint = True, find_best = True, save_NN = False, plot = True):
    
    '''
    input_patches: the number of patches of input large scales
    '''
    
    if isinstance(data_dir, str):
        Thr, Ls = np.load(data_dir)[:, 0:input_patches]
    else:
        Thr, Ls = data_dir

    Ls_rescaled = np.zeros_like(Ls)
    for i in range(Ls.shape[0]):

            Ls_rescaled[i] = rescale_min_max(Ls[i])

    Ls_rescaled = Ls_rescaled.reshape((Ls.shape[0], Ls.shape[1], Ls.shape[1], 1)) 

    rhos_t, f_t, u_t, chi_t = [], [], [], []
    # npatches = 174
    for i in range(0,input_patches):

        mT = rescale_min_max(Thr[i], return_min_max=False)
        rhos_T, f_T, u_T, chi_T= get_functionals_fix(mT)

        f_t.append(f_T);  u_t.append(u_T); chi_t.append(chi_T)

    f_t = np.array(f_t); u_t = np.array(u_t); chi_t = np.array(chi_t)
    
    def get_one_MF_only_NN(Ls_rescaled, k, checkpoint, find_best = False, save_NN = False):
        '''
        need to define checkpoint first;
        k: the epoch of saved model;
        find_best: whether restore new checkpoints(True) or not(False);
        '''
        
        if find_best:
            checkpoint.restore(model_dir + 'training_checkpoints/ckpt-%s'%k)
            
        NNout = checkpoint.generator.predict(Ls_rescaled)
        # print('NNout.shape:', NNout.shape)
        
        if save_NN:
            np.save(save_NN, NNout)

        rhos_nn, f_nn, u_nn, chi_nn = [], [], [], []  
        # if k % 10 == 0:
        #     print(k)
        for i in range(0,input_patches):

            mNN = rescale_min_max(NNout[i,:,:,0], return_min_max=False)
            rhos_NN, f_NN, u_NN, chi_NN= get_functionals_fix(mNN)

            f_nn.append(f_NN); u_nn.append(u_NN);chi_nn.append(chi_NN); 

        f_nn = np.array(f_nn); u_nn = np.array(u_nn); chi_nn = np.array(chi_nn); 

        m1_nnq = compute_intersection(rhos_T, 
                         [np.mean(f_t, axis=0)-np.std(f_t, axis=0), np.mean(f_t, axis=0)+np.std(f_t, axis=0)], 
                         [np.mean(f_nn, axis=0)-np.std(f_nn, axis=0),np.mean(f_nn, axis=0)+np.std(f_nn, axis=0)], 
                         npt=100000)
        m2_nnq = compute_intersection(rhos_T, 
                             [np.mean(u_t, axis=0)-np.std(u_t, axis=0), np.mean(u_t, axis=0)+np.std(u_t, axis=0)], 
                             [np.mean(u_nn, axis=0)-np.std(u_nn, axis=0),np.mean(u_nn, axis=0)+np.std(u_nn, axis=0)], 
                             npt=100000)
        m3_nnq = compute_intersection(rhos_T, 
                             [np.mean(chi_t, axis=0)-np.std(chi_t, axis=0), np.mean(chi_t, axis=0)+np.std(chi_t, axis=0)], 
                             [np.mean(chi_nn, axis=0)-np.std(chi_nn, axis=0),np.mean(chi_nn, axis=0)+np.std(chi_nn, axis=0)], 
                             npt=100000)
        return f_nn, u_nn, chi_nn, m1_nnq, m2_nnq, m3_nnq
    
    if not os.path.isfile(MF_dir):
        with open(MF_dir, "a") as o:

            for k in range(1, 400):
                f_nn, u_nn, chi_nn, m1_nnq, m2_nnq, m3_nnq = get_one_MF_only_NN(Ls_rescaled, k, checkpoint, find_best)
                o.write('%d %.2f %.2f %.2f\n'%(k, m1_nnq, m2_nnq, m3_nnq))
    if best_epoch:
        best_epoch = best_epoch
    else:
        best_epoch = plot_Overlapping(MF_dir, plot)
    
    f_nn, u_nn, chi_nn, m1_nnq, m2_nnq, m3_nnq = get_one_MF_only_NN(Ls_rescaled, best_epoch, checkpoint, find_best, save_NN)
    # print('%.2f %.2f %.2f'%(m1_nnq, m2_nnq, m3_nnq))
    return m1_nnq, m2_nnq, m3_nnq, rhos_T, f_t, u_t, chi_t, f_nn, u_nn, chi_nn, best_epoch
    
def plot_Overlapping(filename, plot = True):
    data = np.loadtxt(filename)
    nepoches = len(data[:, 0])
    epoch = np.arange(1, nepoches + 1)
    mf_mean = np.mean(data[:, 1:4], axis = 1)
    index = np.argsort(mf_mean) + 1
    print('{%s, %0.3f}'%(index[-1], mf_mean[index[-1] - 1]), '{%s, %0.3f}'%(index[-2], mf_mean[index[-2] - 1]))
    print(data[index[-1]-1, 1:4], data[index[-2]-1, 1:4])
          
    if plot:
        plt.figure(figsize = (30,8))
        plt.scatter(epoch, data[:, 1],label = 'v1')
        plt.scatter(epoch, data[:, 2], label = 'v2')
        plt.scatter(epoch, data[:, 3], label = 'v3')
        plt.legend(fontsize = 25)
        plt.xticks(np.arange(1, nepoches, 10))
        plt.grid()
    return index[-1]

def plot_MF(results, S, savedir = False):
    rhos_Y, f_t, u_t, chi_t, f_nn, u_nn, chi_nn = results[3:10] 
    fig, axes = plt.subplots(1,3, figsize=(24, 4))

    for i in range(3):
        f_nn = results[7+i]; f_t = results[4+i]
        axes[i].fill_between(rhos_Y, 
                             np.mean(f_nn, axis=0)-np.std(f_nn, axis=0), 
                             np.mean(f_nn, axis=0)+np.std(f_nn, axis=0), 
                             lw=1, label=r'$m_{ss}^{NN, %s}$'%S[0], alpha=0.5, color='#F87217')
        axes[i].plot(rhos_Y, np.mean(f_nn, axis=0), lw=3, ls='--', color='#D04A00')
        axes[i].fill_between(rhos_Y, 
                             np.mean(f_t, axis=0)-np.std(f_t, axis=0), 
                             np.mean(f_t, axis=0)+np.std(f_t, axis=0), 
                             lw=2, label = r'$m_{ss}^{real, I}$', edgecolor='black', facecolor='None')
        axes[i].plot(rhos_Y, np.mean(f_t, axis=0), lw=2, ls='--', color='black')
        
        axes[i].set_xlabel(r'$\rho$', fontsize=20)
        axes[i].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s'%(i, S), fontsize=20)
        axes[i].set_title('%.2f'%results[i], fontsize = 20)
        if i == 0:
            axes[i].legend(fontsize = 25)
    if savedir:
        plt.savefig(savedir, format = 'pdf')