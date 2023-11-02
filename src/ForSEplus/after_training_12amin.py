import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import healpy as hp
import pymaster as nmt

from .utility import rescale_min_max, compute_intersection, get_functionals_fix # fix the bug of calculating the MFs

### load output (small scales only at 12amin) from the neural network 
# :shape: (174, 320, 320)

class post_training(object):

    
    def __init__(self, NNout_Q, NNout_U, ss_I, Ls_Q_80amin, Ls_U_80amin, MF = True, patch_id = False, fix_MF = True):
        '''
        All processes after the training.

        Parameters
        ----------

        NNout_Q/U: ndarray
            small scales from NN, with shape (174, 320, 320)
        ss_I: ndarray
            intensity small scales 
        Ls_Q/U:  ndarray
            input training files for the NN, shape (2, 348, 320, 320) in units of uK. 
        MF: Bool
        fix_MF: Bool
            Set it to be `True` to use the corrected algorithm to calculate the Minkowski functionals.
        patch_id: focus on only one patch
        '''        
        
        if patch_id is not False:
            assert NNout_Q.shape == (1, 320, 320, 1), "shape should be (320, 320, 1)"
            
            self.NNout_Q = np.copy(NNout_Q).reshape(1, 320, 320);
            self.NNout_U = np.copy(NNout_U).reshape(1, 320, 320);

            self.thr = ss_I; # intensity small scales at 12amin or 3amin
            self.Ls_Q = Ls_Q_80amin[patch_id].reshape(1, 320, 320)
            self.Ls_U = Ls_U_80amin[patch_id].reshape(1, 320, 320)
            self.patch_id = patch_id

        else:
            self.NNout_Q = np.copy(NNout_Q);
            self.NNout_U = np.copy(NNout_U);
            self.thr = ss_I; # intensity small scales at 12amin or 3amin
            self.Ls_Q = Ls_Q_80amin
            self.Ls_U = Ls_U_80amin
            self.patch_id = patch_id
            
        self.fix_MF = fix_MF
        
        if MF:
            self.MF_I = self.get_one_MF(self.thr, npatches = 174, patch_N = False)
            
    def first_normalization(self, gauss_ss_mean_std):
        '''
        Normalize the small scales w.r.t. Gaussian case in the map level;
        
        Parameters
        ----------
        maps_out_12Q/U: ndarray
            small scales generated from the Neural Network. With shape: (174, 320, 320)
        gauss_ss_mean_std: ndarray
            mean and std for each patch of small scales of Gaussian realization, defined by the ratio: 
            `Gaussian_maps_12amin/Gaussian_maps_80amin`. Including 4 files\: Q_mean, Q_std, U_mean, U_std. With shape (4, 174)
        
        Returns
        -------
        
        normalized maps.

        '''
        
        NNout_normed_Q, NNout_normed_U = self.NNout_Q, self.NNout_U
        
        if self.patch_id is not False:
            N_patch = 1
            gauss_ss_mean_std_in = gauss_ss_mean_std[:, self.patch_id:(self.patch_id+1)]
            
        else:
            N_patch = 174
            gauss_ss_mean_std_in = gauss_ss_mean_std
            
        for i in range(N_patch):
            NNout_normed_Q[i] = NNout_normed_Q[i]/np.std(NNout_normed_Q[i])*gauss_ss_mean_std_in[1][i]
            NNout_normed_Q[i] = NNout_normed_Q[i]-np.mean(NNout_normed_Q[i])+gauss_ss_mean_std_in[0][i]
            NNout_normed_U[i] = NNout_normed_U[i]/np.std(NNout_normed_U[i])*gauss_ss_mean_std_in[3][i]
            NNout_normed_U[i] = NNout_normed_U[i]-np.mean(NNout_normed_U[i])+gauss_ss_mean_std_in[2][i]
    
        return NNout_normed_Q, NNout_normed_U

    def normalization(self, gauss_ss_ps, gauss_ss_mean_std, mask_path = 'mask_320x320.npy', lmax = 1000, save_path = False, ss_only = False):
        '''
        Normalize the small scales w.r.t. Gaussian case in the power spectra level and multiply with the large scales to get a full-resolution maps, after the first normalization.
        
        Parameters
        ----------
        
        maps_out_12Q/U:ndarray
            small scales after the first normalization; With shape: (174, 320, 320);
        gauss_ss_ps: ndarray
            power spectra for each patch of small scales of Gaussian realization; 2 in 1: cl_QQ and cl_UU; with shape: (2, 174, 1, 25).
        Ls_Q/U: ndarray
            large scales, same as the input for the training; with shape (174,320,320).
        
        Returns
        -------
        
        patches of full resolution maps with physical units.
        '''
        maps_out_3Q, maps_out_3U = self.first_normalization(gauss_ss_mean_std)
        # print(maps_out_3Q.shape)
        Lx = np.radians(20.); Ly = np.radians(20.)
        Nx = 320; Ny = 320

        mask = np.load(mask_path)

        l0_bins = np.arange(20, lmax, 40)
        lf_bins = np.arange(20, lmax, 40)+39
        b = nmt.NmtBinFlat(l0_bins, lf_bins)
        ells_uncoupled = b.get_effective_ells()

        f_SSQ = nmt.NmtFieldFlat(Lx, Ly, mask, [np.zeros((320, 320))])
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f_SSQ, f_SSQ, b)
        
        if self.patch_id is not False:
            
            N_patch = 1
            gauss_ss_mean_std_in = gauss_ss_mean_std[:, self.patch_id:(self.patch_id+1)]
            gauss_ss_ps_in = gauss_ss_ps[:, self.patch_id:(self.patch_id+1), :]
            
        else:
            N_patch = 174
            gauss_ss_mean_std_in = gauss_ss_mean_std
            gauss_ss_ps_in = gauss_ss_ps
            
        NNmapQ_corr = np.ones((N_patch, 320, 320))
        NNmapU_corr = np.ones((N_patch, 320, 320))
        
        if ss_only:
            NNmapQ_corr_ssonly = np.ones((N_patch, 320, 320))
            NNmapU_corr_ssonly = np.ones((N_patch, 320, 320))    
            
        for i in range(0, N_patch):
            
            f_NNQ = nmt.NmtFieldFlat(Lx, Ly, mask, [maps_out_3Q[i]])
            cl_NN_coupledQ = nmt.compute_coupled_cell_flat(f_NNQ, f_NNQ, b)
            cl_NN_uncoupledQ = w00.decouple_cell(cl_NN_coupledQ)
            f_NNU = nmt.NmtFieldFlat(Lx, Ly, mask, [maps_out_3U[i]])
            cl_NN_coupledU = nmt.compute_coupled_cell_flat(f_NNU, f_NNU, b)
            cl_NN_uncoupledU = w00.decouple_cell(cl_NN_coupledU)

            newQ = maps_out_3Q[i]/np.sqrt(np.mean(cl_NN_uncoupledQ[0][4:]/gauss_ss_ps_in[0][i][0][4:]))
            newU = maps_out_3U[i]/np.sqrt(np.mean(cl_NN_uncoupledU[0][4:]/gauss_ss_ps_in[1][i][0][4:]))
            
            if ss_only:
                NNmapQ_corr_ssonly[i] = ((newQ)-np.mean(newQ)+gauss_ss_mean_std_in[0][i])
                NNmapU_corr_ssonly[i] = ((newU)-np.mean(newU)+gauss_ss_mean_std_in[2][i])
                
            NNmapQ_corr[i] = ((newQ)-np.mean(newQ)+gauss_ss_mean_std_in[0][i])*self.Ls_Q[i]
            NNmapU_corr[i] = ((newU)-np.mean(newU)+gauss_ss_mean_std_in[2][i])*self.Ls_U[i]
            
        self.NNmapQ_corr, self.NNmapU_corr = NNmapQ_corr, NNmapU_corr
        
        if ss_only:
            self.NNmapQ_corr_ssonly, self.NNmapU_corr_ssonly = NNmapQ_corr_ssonly, NNmapU_corr_ssonly
        # return NNmapQ_corr, NNmapU_corr
    
        if save_path:
            np.save(save_path[0], NNmapQ_corr) # .replace('U', '')
            np.save(save_path[1], NNmapU_corr)
            
    def get_one_MF(self, input_maps, npatches = 174, patch_N = False):
        '''
        Defined for output at 12amin, [174, 320, 320] or for ordinary maps with shape [174, 320, 320]
        for nn output at 12amin, npatches = 174; for intensity small scales, npatch = 174;
        
        Parameters
        ----------      
        
        Returns
        -------

        rhos: ndarray
            threshold values, normally [-1, 1]
        f:
        u:
        chi: three kinds of MFs for each patch
        
        '''
        rhos, f_all, u_all, chi_all = [], [], [], []
        maps_MF = input_maps # for intensity small scales with shape (174, 320, 320)
        
        if self.fix_MF:
            get_functionals = get_functionals_fix
        else:
            raise Exception("You should set self.fix_MF = True")
        
        for i in range(0,npatches):
    
            mT = rescale_min_max(maps_MF[i], return_min_max=False)
            rhos, f, u, chi= get_functionals(mT)

            f_all.append(f);  u_all.append(u); chi_all.append(chi)

        f_all = np.array(f_all); u_all = np.array(u_all); chi_all = np.array(chi_all)
        
        return rhos, f_all, u_all, chi_all
    
    def compute_overlapping(self, MF_P):
        '''
        
        Parameters
        ----------      
        
        compute the MFs overlapping fraction w.r.t. small scales of intensity maps
        
        
        Returns
        -------
        '''
        
        rhos_T, f_t, u_t, chi_t = self.MF_I;
        rho_NN, f_nn, u_nn, chi_nn = MF_P;
        
        m1_nnq = compute_intersection(rhos_T, 
                         [np.mean(f_t, axis=0)-np.std(f_t, axis=0), np.mean(f_t, axis=0)+np.std(f_t, axis=0)], 
                         [np.mean(f_nn, axis=0)-np.std(f_nn, axis=0),np.mean(f_nn, axis=0)+np.std(f_nn, axis=0)], npt=100000)
        m2_nnq = compute_intersection(rhos_T, 
                         [np.mean(u_t, axis=0)-np.std(u_t, axis=0), np.mean(u_t, axis=0)+np.std(u_t, axis=0)], 
                         [np.mean(u_nn, axis=0)-np.std(u_nn, axis=0),np.mean(u_nn, axis=0)+np.std(u_nn, axis=0)], npt=100000)
        m3_nnq = compute_intersection(rhos_T, 
                         [np.mean(chi_t, axis=0)-np.std(chi_t, axis=0), np.mean(chi_t, axis=0)+np.std(chi_t, axis=0)], 
                         [np.mean(chi_nn, axis=0)-np.std(chi_nn, axis=0),np.mean(chi_nn, axis=0)+np.std(chi_nn, axis=0)], npt=100000)
        
        return m1_nnq, m2_nnq,m3_nnq
    
    def plot_MF(self, savedir = False, save_format = False):
        '''  

        Parameters
        ----------      

        Returns
        -------

        '''

        rhos_Y, f_t, u_t, chi_t = self.MF_I
        MF_Q = self.get_one_MF(self.NNout_Q);
        MF_U = self.get_one_MF(self.NNout_U);

        results = [];
        results.append(self.compute_overlapping(MF_Q))
        results.append(self.compute_overlapping(MF_U))

        rhos_Y, f_nn_Q, u_nn_Q, chi_nn_Q = MF_Q
        rhos_Y, f_nn_U, u_nn_U, chi_nn_U = MF_U    

        f_nn_all = [[f_nn_Q, u_nn_Q, chi_nn_Q],[f_nn_U, u_nn_U, chi_nn_U]]
        f_i = [f_t, u_t, chi_t]
        fig, axes = plt.subplots(2,3, figsize=(24, 10))
        S = ['Q', 'U']
        for i in range(3):
            for j in range(2):

                axes[j, i].ticklabel_format(axis='y', style='sci', scilimits=(4, 4))

                if i == 0:
                    label_nn = r'$\tilde{m}^{%s, 20^{\circ}}_{12^\prime}$'%S[j]
                    label_ii = r'$\tilde{m}^{I, 20^{\circ}}_{12^\prime}$'
                else:
                    label_nn = None
                    label_ii = None

                f_nn = f_nn_all[j][i]; f_t = f_i[i];

                axes[j, i].fill_between(rhos_Y, 
                                     np.mean(f_nn, axis=0)-np.std(f_nn, axis=0), 
                                     np.mean(f_nn, axis=0)+np.std(f_nn, axis=0), 
                                     lw=1, label = label_nn, alpha=0.5, color='#F87217')
                axes[j, i].plot(rhos_Y, np.mean(f_nn, axis=0), lw=3, ls='--', color='#D04A00')
                axes[j, i].fill_between(rhos_Y, 
                                     np.mean(f_t, axis=0)-np.std(f_t, axis=0), 
                                     np.mean(f_t, axis=0)+np.std(f_t, axis=0), 
                                     lw=2, label = label_ii, edgecolor='black', facecolor='None')
                axes[j, i].plot(rhos_Y, np.mean(f_t, axis=0), lw=2, ls='--', color='black')

                axes[j, i].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s'%(i, S[j]), fontsize=25)

                if j == 1:
                    axes[j, i].set_xlabel(r'$\rho$', fontsize=25)

                axes[j, i].legend(title = 'OF = %.2f'%results[j][i], fontsize = 30, frameon=False, title_fontsize=30)

        plt.tight_layout()
        if savedir:
            plt.savefig(savedir, format = save_format)
 
    def reproject_to_fullsky(self, ):
        
        '''
        salloc --nodes 4 --qos interactive --time 00:30:00 --constraint cpu --account=mp107
        module load tensorflow/2.6.0
        srun -n 16 python reproject2fullsky_mpi.py --pixelsize 3.75 --npix 320 --overlap 2   --verbose  --flat-projection /pscratch/sd/j/jianyao/forse_processed_data/NN_out_Q_12amin_physical_units_from_real_Nico.npy --flat2hpx --nside 2048 --apodization-file /global/homes/j/jianyao/Small_Scale_Foreground/mask_320*320.npy --adaptive-reprojection
        
        srun -n 16 python reproject2fullsky_mpi.py --pixelsize 0.9375 --npix 1280 --overlap 2   --verbose  --flat-projection  /pscratch/sd/j/jianyao/forse_output/Nico_Q_20amin_20x20_1280.npy --flat2hpx --nside 4096 --apodization-file /global/homes/j/jianyao/Small_Scale_Foreground/mask_1280*1280.npy --adaptive-reprojection
        
        '''
        
        pass
    
    def power_spectra_patch(self, n, lmax = 1000, w22_file = "w22_flat_320_320.fits", mask_path = 'mask_320*320.npy', plot = True, return_cls = False, save_path = False):
        
        '''
        plot EE/BB power spectra for each flat patch of sky. For Large scales only, Large scales with gaussian small scales; 
        Large scales with ForSE small scales. 
        '''
        
        Lx = np.radians(20.); Ly = np.radians(20.)
        Nx = 320; Ny = 320

        gaussian_pathQ ='/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/GNILC_gaussian_ss_Q_20x20deg_Npix320_full_sky_adaptive.npy'
        gaussian_pathU ='/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/GNILC_gaussian_ss_U_20x20deg_Npix320_full_sky_adaptive.npy'
        
        gaussian_mapsQ = np.load(gaussian_pathQ, allow_pickle = True)*1e6
        gaussian_mapsU = np.load(gaussian_pathU, allow_pickle = True)*1e6
        
        mask = np.load(mask_path)
        l0_bins = np.arange(20, lmax, 40); lf_bins = np.arange(20, lmax, 40)+39
        b = nmt.NmtBinFlat(l0_bins, lf_bins)
        ells_uncoupled = b.get_effective_ells()
        
        w22 = nmt.NmtWorkspaceFlat()
        try:
            w22.read_from(w22_file)
            print('weights loaded from %s' % w22_file)
        except:
            
            f_2 = nmt.NmtFieldFlat(Lx, Ly, mask, [np.zeros((320, 320)), np.zeros((320, 320))], purify_b=True)
            w22.compute_coupling_matrix(f2, f2, b)
            w22.write_to(w22_file)
            print('weights writing to disk')
        
        Qmaps = [self.Ls_Q[n], gaussian_mapsQ[n], self.NNmapQ_corr[n]];
        Umaps = [self.Ls_U[n], gaussian_mapsU[n], self.NNmapU_corr[n]];
        
        cls_all = []
        for i in range(3):

            f_NN = nmt.NmtFieldFlat(Lx, Ly, mask, [Qmaps[i], Umaps[i]], purify_b=True)
            cl_NN_coupled = nmt.compute_coupled_cell_flat(f_NN, f_NN, b)
            cl_NN_uncoupled = w22.decouple_cell(cl_NN_coupled)
            cls_all.append(cl_NN_uncoupled)    
            
        if plot:
            fig, axes = plt.subplots(1,2, figsize=(17, 5.5))                  
            names = ['EE', 'BB']
            for i in range(2):
                axes[i].loglog(ells_uncoupled, cls_all[0][i*3],  '--', lw=2, color='Black', alpha=0.5, label = 'GNILC 80 amin')
                axes[i].loglog(ells_uncoupled, cls_all[1][i*3], '-', label='GNILC+Gauss 12 amin', lw=4, color='#569A62', alpha=0.7)
                axes[i].loglog(ells_uncoupled, cls_all[2][i*3], '-', label='GNILC+NN 12 amin', lw=4, color='#F87217', alpha=0.7)
                axes[i].set_ylim(1e-6, 2e-1)
                axes[i].set_xticks([40, 100, 400, 1000])
                axes[i].set_title('%s'%names[i], fontsize=18)
                axes[i].set_xlabel(r'Multipole $\ell$', fontsize=18)
                axes[i].set_ylabel(r'$C_\ell$ [$\mu K^2$]', fontsize=18)  

            axes[0].legend(fontsize = 15)

            if save_path:
                plt.savefig(save_path, format = 'pdf')
            
        if return_cls:
            return cls_all