import numpy as np
import healpy as hp
from utility import cl_nmt, plot_spectra, cl_anafast
import matplotlib.pyplot as plt

class validate_3amin(object):
    
    def __init__(self, maps_80amin, input_3amin, poltens_3amin):
        '''
        maps only contain QU
        '''
        
        self.maps_80amin = maps_80amin
        self.f_3amin = input_3amin
        self.p_3amin = poltens_3amin
        
        
    def show_fullsky(self):
        fig1 = plt.figure(figsize = (8, 7))
        hp.mollview(self.maps_80amin[0], sub = 321, title = 'GNILC Q', norm = 'hist', cmap='coolwarm', min = -1000, max = 1000, cbar=False)
        hp.mollview(self.maps_80amin[1], sub = 322, title = 'GNILC U', norm = 'hist', cmap='coolwarm', min = -1000, max = 1000, cbar=False)
        hp.mollview(self.f_3amin[0], sub = 323, title = 'ForSE Q 3amin', norm = 'hist', cmap='coolwarm', min = -1000, max = 1000, cbar=False)
        hp.mollview(self.f_3amin[1], sub = 324, title = 'ForSE U 3amin', norm = 'hist', cmap='coolwarm', min = -1000, max = 1000, cbar=False)
        hp.mollview(self.p_3amin[0] - self.maps_80amin[0], sub = 325, title = 'GNILC Q - Q 3amin', norm = 'hist', cmap='coolwarm', min = -1000, max = 1000)
        hp.mollview(self.p_3amin[1] - self.maps_80amin[1], sub = 326, title = 'GNILC Q - Q 3amin', norm = 'hist', cmap='coolwarm', min = -1000, max = 1000)
        
    def show_patch(self, center = [0,-60], cmap = 'coolwarm'):
        '''
        BK center is [0, -60]
        '''
        
        fig1 = plt.figure(figsize = (18, 5))
        hp.gnomview(self.maps_80amin[0],reso=8, xsize=450,ysize=250, coord=['G', 'C'] ,rot=center, sub = 131, cmap = cmap, title = 'Q_GNILC_unires_80amin')
        hp.gnomview(self.f_3amin[0],reso=8, xsize=450,ysize=250, coord=['G', 'C'] ,rot=center, sub = 132, min = -30.3, max = 164, cmap = cmap, title = 'Forse_3amin', notext = True)
        hp.gnomview(self.p_3amin[0],reso=8, xsize=450,ysize=250, coord=['G', 'C'] ,rot=center, sub = 133, min = -30.3, max = 164, cmap = cmap, title = 'Poltens_3amin', notext = True)

        fig1 = plt.figure(figsize = (18, 5))
        hp.gnomview(self.maps_80amin[1],reso=8, xsize=450,ysize=250, coord=['G', 'C'] ,rot=center, sub = 131, cmap = cmap, title = 'U_GNILC_unires_80amin')
        hp.gnomview(self.f_3amin[1],reso=8, xsize=450,ysize=250, coord=['G', 'C'] ,rot=center, sub = 132, min = -86.2, max = 71.3, cmap = cmap, title = 'Forse_3amin', notext = True)
        hp.gnomview(self.p_3amin[1],reso=8, xsize=450,ysize=250, coord=['G', 'C'] ,rot=center, sub = 133, min = -86.2, max = 71.3, cmap = cmap, title = 'Poltens_3amin', notext = True)
        
    def cls_check_sky(self, maps, planck_mask, lmax = 4096, color = 'r', label = 'ForSE-3\'', cls_pt = True):
        
        maps_3amin_2048 = np.array((hp.ud_grade(maps[0], nside_out = 2048), hp.ud_grade(maps[1], nside_out = 2048)))

        ells_full_3amin, cl_full_3amin = cl_anafast(maps_3amin_2048, lmax = lmax)
        cl_full_3amin_na = [cl_full_3amin[1], cl_full_3amin[4], cl_full_3amin[4], cl_full_3amin[2]]

        w22_file_80p = '/pscratch/sd/j/jianyao/w22_2048_80_sky_lmax_4096_nbins_40.fits'
        ell_80p_3amin, cl_80p_3amin = cl_nmt(2048, planck_mask[4], maps_3amin_2048,lmax = lmax, nlbins = 40, w22_file = w22_file_80p)

        w22_file_40p = '/pscratch/sd/j/jianyao/w22_2048_40_sky_lmax_4096_nbins_40.fits'
        ell_40p_3amin, cl_40p_3amin = cl_nmt(2048, planck_mask[1], maps_3amin_2048,lmax = lmax, nlbins = 40, w22_file = w22_file_40p)
        
        cls_full_3amin = {'ells':ells_full_3amin, 'spectra':cl_full_3amin_na, 'color':'%s-'%color, 'label':label}
        cls_80p_3amin = {'ells':ell_80p_3amin, 'spectra':cl_80p_3amin, 'color':'%s-.'%color, 'label':''}
        cls_40p_3amin = {'ells':ell_40p_3amin, 'spectra':cl_40p_3amin, 'color':'%s--'%color, 'label':''}
        
        self.cls_maps = [cls_full_3amin, cls_80p_3amin, cls_40p_3amin]
        # cls_full_pt, cls_80p_pt, cls_40p_pt
        cls_all = [cls_pt[0], cls_pt[1], cls_pt[2], cls_full_3amin, cls_80p_3amin, cls_40p_3amin]
        names = ['EE', 'BB']
        fig1 = plot_spectra(cls_all, names, save_dir = False,lim = [1e-8, 5e4])
    
    def cls_check_patch(self, maskbk):
        
        maps_80amin_512 = np.array((hp.ud_grade(self.maps_80amin[0], nside_out = 512), hp.ud_grade(self.maps_80amin[1], nside_out = 512)))
        maps_3amin_512 = np.array((hp.ud_grade(self.f_3amin[0], nside_out = 512), hp.ud_grade(self.f_3amin[1], nside_out = 512)))
        poltens_3amin_512 =  np.array((hp.ud_grade(self.p_3amin[0], nside_out = 512), hp.ud_grade(self.p_3amin[1], nside_out = 512)))

        ells_uncoupled, cl_80amin_bk = cl_nmt(512, maskbk, maps_80amin_512, 1000, 20, dl=True, w22_file = '/pscratch/sd/j/jianyao/w22_512_bicep.fits')
        ells_uncoupled, cl_3amin_bk = cl_nmt(512, maskbk, maps_3amin_512, 1000, 20, dl=True, w22_file = '/pscratch/sd/j/jianyao/w22_512_bicep.fits')
        ells_uncoupled, cl_3amin_pt_bk = cl_nmt(512, maskbk, poltens_3amin_512, 1000, 20, dl=True, w22_file = '/pscratch/sd/j/jianyao/w22_512_bicep.fits')
        
        def model(ell, A, gamma):
            out = A * (ell/80) ** gamma
            return out
        
        fig1, axes = plt.subplots(1,2, figsize = (18, 6))                
        names = ['EE', 'BB']
        for i in range(2):
            if i == 0:
                axes[i].loglog(ells_uncoupled, cl_80amin_bk[i*3],  '--', lw=3, color='Black', alpha=0.5, label = 'GNILC 80 amin')
                # axes[i].loglog(ells_uncoupled, cl_12amin_bk[i*3], '--', label='GNILC+ NN 12 amin', lw=3, color='#569A62', alpha=0.5)
                axes[i].loglog(ells_uncoupled, cl_3amin_bk[i*3], '-', label='ForSE 3 amin', lw=3, color='red', alpha=0.7)
                axes[i].loglog(ells_uncoupled, cl_3amin_pt_bk[i*3], '-', label='poltens 3amin', lw=3, color='green', alpha=0.7)
            else:
                axes[i].loglog(ells_uncoupled, cl_80amin_bk[i*3],  '--', lw=3, color='Black', alpha=0.5)
                # axes[i].loglog(ells_uncoupled, cl_12amin_bk[i*3], '--', lw=3, color='#569A62', alpha=0.5)
                axes[i].loglog(ells_uncoupled, cl_3amin_bk[i*3], '-', lw=3, color='red', alpha=0.7) ##F87217
                axes[i].loglog(ells_uncoupled, cl_3amin_pt_bk[i*3], '-', lw=3, color='green', alpha=0.7)       

            axes[i].set_ylim(1e-1, 1e2)
            axes[i].set_xticks([40, 100, 400, 1000])
            axes[i].set_title('%s'%names[i], fontsize=18)
            axes[i].set_xlabel(r'Multipole $\ell$', fontsize=18)
            axes[i].set_ylabel(r'$D_\ell$ [$\mu K_{CMB}^2$]', fontsize=18)  
            axes[i].set_xlim(40,1e3) 

        axes[0].legend(fontsize = 15, loc = 'lower left')
        axes[1].loglog(ells_uncoupled, model(ells_uncoupled, 4.25, -0.4), 'b--', label = r'BICEP/KECK power law, $\alpha$ = - 0.4', lw = 3)
        axes[0].loglog(ells_uncoupled, model(ells_uncoupled, 4.25, -0.4)*2, 'b--', label = 'BICEP/KECK power law', lw = 3)
        axes[1].scatter([80.], [4.25], color='k', marker='x',  label=r'$A_d$_BK2021 @353GHz = 4.25 ${\mu K}^2$')
        axes[1].legend(fontsize = 15, loc = 'lower left')
        
        return ells_uncoupled, cl_80amin_bk, cl_3amin_bk, cl_3amin_pt_bk
    
    def show_EB_ratio(self, ells, cls_p, cls_f, f = '80%'):
        
        fig1 = plt.figure()
        plt.plot(ells,  cls_p[0]/cls_p[3], label = 'poltens-3amin', color = 'green')
        plt.plot(ells,  cls_f[0]/cls_f[3], label = 'Forse-3amin', color = 'red')
        plt.axhline(2.0, ls  = '--', color = 'blue')
        plt.ylim(1,2.5)
        plt.legend()
        plt.title('E/B ratio, %s sky'%f)

        
    