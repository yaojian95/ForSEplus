import numpy as np
import healpy as hp
import scipy.ndimage
from numba import jit
import pymaster as nmt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def rescale_input(Ls, random_noise = None):
    Ls_rescaled = np.zeros_like(Ls)
    
    if random_noise is not None:
        assert random_noise.shape == Ls.shape
        
        for i in range(Ls.shape[0]):
            Ls_rescaled[i] = rescale_min_max(rescale_min_max(Ls[i]) + random_noise[i])
            
    else:   
        for i in range(Ls.shape[0]):
                Ls_rescaled[i] = rescale_min_max(Ls[i])
                
    Ls_rescaled = Ls_rescaled.reshape((Ls.shape[0], Ls.shape[1], Ls.shape[1], 1)) 
    return Ls_rescaled

def sigmoid(x, x0, width, power=4):
    """Sigmoid function given start point and width
    Parameters
    ----------
    x : array
        input x axis
    x0 : float
        value of x where the sigmoid starts (not the center)
    width : float
        width of the transition region in unit of x
    power : float
        tweak the steepness of the curve
    Returns
    -------
    sigmoid : array
        sigmoid, same length of x"""
    return 1.0 / (1 + np.exp(-power * (x - x0 - width / 2) / width))

def correct_EB(full_Q, full_U, reso = '3amin'):
    '''
    set E/B ratio to 2;
    
    Return
    ------
    Q and U maps
    '''
    
    if reso == '3amin':
        lmax = 4096; nside_out = 4096
        
    elif reso == '12amin':
        lmax = 2048; nside_out = 2048
        
    mapI_3amin = np.ones_like(full_Q)
    maps_3amin = np.array((mapI_3amin, full_Q, full_U))
    alms = hp.map2alm(maps_3amin, lmax = lmax)
    factors_full = 1 - sigmoid(np.arange(lmax + 10), 141.5, 400)*0.5
    alms[2] = hp.almxfl(alms[2], np.sqrt(factors_full))
    maps_new = hp.alm2map(alms, nside = nside_out)
    
    return maps_new[1:]

def rescale_min_max(img, a=-1, b=1, return_min_max=False):
    img_resc = (b-a)*(img-np.min(img))/(np.max(img)-np.min(img))+a
    if return_min_max:
        return img_resc, np.min(img), np.max(img)
    else:
        return img_resc
    
def upsampling(maps_ren2_12Q, maps_ren2_12U):
    maps_upx_12Q = np.zeros([174,1280,1280])
    maps_upx_12U = np.zeros([174,1280,1280])

    for i in range(174):
        maps_upx_12Q[i] = maps_ren2_12Q[i].repeat(4, axis = 1).repeat(4, axis = 0)
        maps_upx_12U[i] = maps_ren2_12U[i].repeat(4, axis = 1).repeat(4, axis = 0)

    return maps_upx_12Q, maps_upx_12U

def smoothing_20(maps_upx_12Q, maps_upx_12U, sigma = 1.81):
    '''
    smoothing with a Gaussian kernal for the pixelized image.
    '''
    
    maps_smth_20Q = np.zeros([174,1280,1280])
    maps_smth_20U = np.zeros([174,1280,1280])

    for i in range(174):
        maps_smth_20Q[i] = scipy.ndimage.gaussian_filter(maps_upx_12Q[i], sigma = sigma*4, order = 0, mode = 'reflect')
        maps_smth_20U[i] = scipy.ndimage.gaussian_filter(maps_upx_12U[i], sigma = sigma*4, order = 0, mode = 'reflect')

    return maps_smth_20Q , maps_smth_20U

def divide(maps_smth_20Q, maps_smth_20U):
    maps_sub_20Q = np.zeros([174,49,320,320])
    maps_sub_20U = np.zeros([174,49,320,320])

    for i in range(174):
        for j in range(0,1120,160):
            for k in range(0,1120,160):
                maps_sub_20Q[i,int(j/160)*7+int(k/160),:,:] = maps_smth_20Q[i,j:(j+320),k:(k+320)]
                maps_sub_20U[i,int(j/160)*7+int(k/160),:,:] = maps_smth_20U[i,j:(j+320),k:(k+320)]

    return maps_sub_20Q, maps_sub_20U
    
def combine_to_20by20(NNmapQ_corr, NNmapU_corr, save_dir = False):
    '''
    Recompose 5°x5° maps together to form 20°x20° maps
    '''
    # Create apodization masks
    x = np.cos(np.concatenate((np.arange(160)[::-1],np.arange(160)))*np.pi/(2*159))*np.cos(np.concatenate((np.arange(160)[::-1],np.arange(160)))*np.pi/(2*159))
    mask_cc = np.zeros((320,320))
    for i in range(320):
        for j in range (320):
            mask_cc[i,j] = x[i]*x[j] 

    mask_nw = np.copy(mask_cc)
    for i in range(160):
        for j in range(160):
            mask_nw[i,j] = 1
    for i in range(160):
        for j in range(160,320):
            mask_nw[i,j] = x[j]
    for i in range(160,320):
        for j in range(160):
            mask_nw[i,j] = x[i]

    mask_sw = np.rot90(mask_nw); mask_se = np.rot90(mask_sw); mask_ne = np.rot90(mask_se)

    mask_nn = np.copy(mask_cc)
    for i in range(160):
        for j in range(320): 
            mask_nn[i,j] = x[j]

    mask_ww = np.rot90(mask_nn); mask_ss = np.rot90(mask_ww); mask_ee = np.rot90(mask_ss)

    # Apply apodization masks to 5°x5° maps
    maps_msk_3Q = np.zeros(np.shape(NNmapQ_corr));
    maps_msk_3U = np.zeros(np.shape(NNmapU_corr));
    maps_ren2_3Q = NNmapQ_corr;
    maps_ren2_3U = NNmapU_corr;

    for i in range(174):  

        # angle masks
        maps_msk_3Q[i,0,:,:] = maps_ren2_3Q[i,0,:,:]*mask_nw
        maps_msk_3Q[i,6,:,:] = maps_ren2_3Q[i,6,:,:]*mask_ne
        maps_msk_3Q[i,42,:,:] = maps_ren2_3Q[i,42,:,:]*mask_sw
        maps_msk_3Q[i,48,:,:] = maps_ren2_3Q[i,48,:,:]*mask_se

        maps_msk_3U[i,0,:,:] = maps_ren2_3U[i,0,:,:]*mask_nw
        maps_msk_3U[i,6,:,:] = maps_ren2_3U[i,6,:,:]*mask_ne
        maps_msk_3U[i,42,:,:] = maps_ren2_3U[i,42,:,:]*mask_sw
        maps_msk_3U[i,48,:,:] = maps_ren2_3U[i,48,:,:]*mask_se

        # side masks
        for j in range(1,6):
            maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_nn
            maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_nn
        for j in range(43,48):
            maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_ss
            maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_ss
        for j in [7,14,21,28,35]:
            maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_ww
            maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_ww
        for j in [13,20,27,34,41]:
            maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_ee
            maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_ee

        # center masks
        for j in range(8,13):
            maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_cc
            maps_msk_3Q[i,j+7,:,:] = maps_ren2_3Q[i,j+7,:,:]*mask_cc
            maps_msk_3Q[i,j+14,:,:] = maps_ren2_3Q[i,j+14,:,:]*mask_cc
            maps_msk_3Q[i,j+21,:,:] = maps_ren2_3Q[i,j+21,:,:]*mask_cc
            maps_msk_3Q[i,j+28,:,:] = maps_ren2_3Q[i,j+28,:,:]*mask_cc

            maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_cc
            maps_msk_3U[i,j+7,:,:] = maps_ren2_3U[i,j+7,:,:]*mask_cc
            maps_msk_3U[i,j+14,:,:] = maps_ren2_3U[i,j+14,:,:]*mask_cc
            maps_msk_3U[i,j+21,:,:] = maps_ren2_3U[i,j+21,:,:]*mask_cc
            maps_msk_3U[i,j+28,:,:] = maps_ren2_3U[i,j+28,:,:]*mask_cc

    # Recompose 20°x20° maps together
    maps_big_3Q = np.zeros([174,1280,1280])
    maps_big_3U = np.zeros([174,1280,1280])

    for i in range(174): 
        for j in range(0,1120,160):
            for k in range(0,1120,160):
                maps_big_3Q[i,j:(j+320),k:(k+320)] += maps_msk_3Q[i,int(j/160)*7+int(k/160),:,:]
                maps_big_3U[i,j:(j+320),k:(k+320)] += maps_msk_3U[i,int(j/160)*7+int(k/160),:,:]

    NN_20by20_Q = maps_big_3Q;
    NN_20by20_U = maps_big_3U;

    if save_dir:
        np.save(save_dir[0], maps_big_3Q)
        np.save(save_dir[1], maps_big_3U)
    return NN_20by20_Q, NN_20by20_U
    

def bin_array(array, bins=100):
    len_data = len(array)
    x = np.arange(len_data)+1
    num_bin = len_data//bins
    data_binned = []
    x_binned = []
    for i in range(num_bin):
        data_binned.append(np.mean(array[i*bins:(i+1)*bins]))
        x_binned.append(np.mean(x[i*bins:(i+1)*bins]))
    data_binned = np.array(data_binned)
    x_binned = np.array(x_binned)
    return x_binned, data_binned

@jit(nopython=True) 
def estimate_marchingsquare_fix(data , threshold ):
    width = data.shape[0]
    height= data.shape[1]
    f,u,chi=0 ,0,0
    for i in range(width-1 ):
        for j in range(height-1):
            pattern=0
            if (data[i,j]     > threshold) : pattern += 1;
            if (data[i+1,j]   > threshold) : pattern += 2;
            if (data[i+1,j+1] > threshold) : pattern += 4;
            if (data[i,j+1 ]  > threshold) : pattern += 8;
            if pattern ==0 :
                continue
            elif pattern==1:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j])
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,(j+1)]);
                f = f + 0.5 * a1 * a4;
                u = u + np.sqrt(a1 * a1 + a4 * a4);
                chi = chi + 0.25;
                continue;
            elif pattern==2:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1, (j+1)]);
                f = f + 0.5 * (1 - a1) * (a2);
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2);
                chi = chi + 0.25;
                continue;
            elif pattern==3:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,(j+1)]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,(j+1)]);
                f = f + a2 + 0.5 * (a4 - a2);
                u = u + np.sqrt(1 + (a4 - a2) * (a4 - a2));
                continue;
            elif pattern==4:
                a2 = (data[ i+1,j] - threshold) / (data[i+1,j ] - data[ i+1,j+1]);
                a3 = (data[ i,j+1 ] -  threshold) / (data[i,j+1] - data[ i+1,j+1]);
                f = f + 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi + 0.25;
                continue;
            elif pattern==5:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4);
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2) + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi + 0.5;
                continue;
            elif pattern==6:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + (1 - a3) + 0.5 * (a3 - a1);
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1));
                continue;
            elif pattern==7:
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a3 * (1 - a4);
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi - 0.25;
                continue;

            elif pattern==8:
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 0.5 * a3 * (1 - a4);
                u = u + np.sqrt(a3 * a3 + (1 - a4) * (1 - a4));
                chi = chi + 0.25;
                continue;
            elif pattern==9:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + a1 + 0.5 * (a3 - a1);
                u = u + np.sqrt(1 + (a3 - a1) * (a3 - a1));
                continue;
            elif pattern==10:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt(a1 * a1 + a4 * a4) + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi + 0.5;
                continue;
            elif pattern==11:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a3 = (data[i,j+1] - threshold) / (data[i,j+1] - data[i+1,j+1]);
                f = f + 1 - 0.5 * (1 - a2) * (1 - a3);
                u = u + np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3));
                chi = chi - 0.25;
                continue;
            elif pattern==12:
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + (1 - a2) + 0.5 * (a2 - a4);
                u = u + np.sqrt(1 + (a2 - a4) * (a2 - a4));
                continue;
            elif pattern==13:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a2 = (data[i+1,j] - threshold) / (data[i+1,j] - data[i+1,j+1]);
                f = f + 1 - .5 * (1 - a1) * a2;
                u = u + np.sqrt((1 - a1) * (1 - a1) + a2 * a2);
                chi = chi - 0.25;
                continue;
            elif pattern==14:
                a1 = (data[i,j] - threshold) / (data[i,j] - data[i+1,j]);
                a4 = (data[i,j] - threshold) / (data[i,j] - data[i,j+1]);
                f = f + 1 - 0.5 * a1 * a4;
                u = u + np.sqrt(a1 * a1 + a4 * a4);
                chi = chi - 0.25;
                continue;
            elif pattern == 15:
                f +=1 ;
                continue;


    return f,u, chi

def get_functionals_fix(im , nevals= 32):
    vmin =im.min() ; vmax=im.max()
    rhos =  np.linspace( vmin,vmax, nevals)
    f= np.zeros_like(rhos)
    u= np.zeros_like(rhos)
    chi= np.zeros_like(rhos)
    for k, rho in np.ndenumerate( rhos) :
        f[k], u[k],chi[k]=  estimate_marchingsquare_fix(im, rho )
    return rhos, f,u,chi

def compute_intersection(x, cont1, cont2, npt=10000):
    ymin1 = cont1[0]
    ymax1 = cont1[1]
    ymin2 = cont2[0]
    ymax2 = cont2[1]
    yMAX = np.max([ymax1, ymax2])+0.1*np.max([ymax1, ymax2])
    yMIN = np.min([ymin1, ymin2])-0.1*np.min([ymin1, ymin2])
    area1 = 0
    area2 = 0
    areaint = 0
    ind_xi = np.random.randint(0, len(x), npt)
    yi = np.random.uniform(yMIN, yMAX, npt)
    for i in range(npt):
        if ymin1[ind_xi[i]]<=yi[i]<=ymax1[ind_xi[i]]:
            area1 += 1
            if ymin2[ind_xi[i]]<=yi[i]<=ymax2[ind_xi[i]]:
                areaint += 1
        elif ymin2[ind_xi[i]]<=yi[i]<=ymax2[ind_xi[i]]:
            area2 += 1
    return areaint/(area1+area2)


def flat_ps(maps, mask_path, w22_file):
    mask = np.load(mask_path)
    l0_bins = np.arange(20, 1000, 40); lf_bins = np.arange(20, 1000, 40)+39
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()
    w22 = nmt.NmtWorkspaceFlat()
    try:
        w22.read_from(w22_file)
    except:
        print('Error: no workspace loaded')

    f_NN = nmt.NmtFieldFlat(Lx, Ly, mask, [maps[0], maps[1]], purify_b=True)
    cl_NN_coupled = nmt.compute_coupled_cell_flat(f_NN, f_NN, b)
    cl_NN_uncoupled = w22.decouple_cell(cl_NN_coupled)
    
    return ells_uncoupled, cl_NN_uncoupled

def from_12to13(maps_ren2_12Q, maps_ren2_12U, only_one = False):
    
    if only_one:
        N_patch = 1
    else:
        N_patch = 174
    maps_upx_12Q = np.zeros([N_patch,1280,1280])
    maps_upx_12U = np.zeros([N_patch,1280,1280])

    for i in range(N_patch):
        maps_upx_12Q[i] = maps_ren2_12Q[i].repeat(4, axis = 1).repeat(4, axis = 0)
        maps_upx_12U[i] = maps_ren2_12U[i].repeat(4, axis = 1).repeat(4, axis = 0)

    maps_smth_20Q = np.zeros([N_patch,1280,1280])
    maps_smth_20U = np.zeros([N_patch,1280,1280])

    sigma = 2.123/4

    for i in range(N_patch):
        maps_smth_20Q[i] = scipy.ndimage.gaussian_filter(maps_upx_12Q[i], sigma = sigma*4, order = 0, mode = 'reflect')
        maps_smth_20U[i] = scipy.ndimage.gaussian_filter(maps_upx_12U[i], sigma = sigma*4, order = 0, mode = 'reflect')

    maps_sub_20Q = np.zeros([N_patch,49,320,320])
    maps_sub_20U = np.zeros([N_patch,49,320,320])

    for i in range(N_patch):
        for j in range(0,1120,160):
            for k in range(0,1120,160):
                maps_sub_20Q[i,int(j/160)*7+int(k/160),:,:] = maps_smth_20Q[i,j:(j+320),k:(k+320)]
                maps_sub_20U[i,int(j/160)*7+int(k/160),:,:] = maps_smth_20U[i,j:(j+320),k:(k+320)]

    maps_sub_20Q_train = maps_sub_20Q.reshape(N_patch*49, 320, 320)
    maps_sub_20U_train = maps_sub_20U.reshape(N_patch*49, 320, 320)
    
    return maps_sub_20Q_train, maps_sub_20U_train

def from_12to20(maps_ren2_12Q, maps_ren2_12U, random_noise = None, only_one = False):
    '''
    random_noise: add random noise at the last step.
    '''
    if only_one:
        N_patch = 1
    else:
        N_patch = 174
        
    maps_upx_12Q = np.zeros([N_patch,1280,1280])
    maps_upx_12U = np.zeros([N_patch,1280,1280])

    for i in range(N_patch):
        maps_upx_12Q[i] = maps_ren2_12Q[i].repeat(4, axis = 1).repeat(4, axis = 0)
        maps_upx_12U[i] = maps_ren2_12U[i].repeat(4, axis = 1).repeat(4, axis = 0)

    maps_smth_20Q = np.zeros([N_patch,1280,1280])
    maps_smth_20U = np.zeros([N_patch,1280,1280])

    sigma = 1.81

    for i in range(N_patch):
        maps_smth_20Q[i] = scipy.ndimage.gaussian_filter(maps_upx_12Q[i], sigma = sigma*4, order = 0, mode = 'reflect')
        maps_smth_20U[i] = scipy.ndimage.gaussian_filter(maps_upx_12U[i], sigma = sigma*4, order = 0, mode = 'reflect')

    maps_sub_20Q = np.zeros([N_patch,49,320,320])
    maps_sub_20U = np.zeros([N_patch,49,320,320])

    for i in range(N_patch):
        for j in range(0,1120,160):
            for k in range(0,1120,160):
                maps_sub_20Q[i,int(j/160)*7+int(k/160),:,:] = maps_smth_20Q[i,j:(j+320),k:(k+320)]
                maps_sub_20U[i,int(j/160)*7+int(k/160),:,:] = maps_smth_20U[i,j:(j+320),k:(k+320)]

    maps_sub_20Q_train = maps_sub_20Q.reshape(N_patch*49, 320, 320)
    maps_sub_20U_train = maps_sub_20U.reshape(N_patch*49, 320, 320)
    
    if random_noise is not None:
        
        assert random_noise.shape == maps_sub_20Q_train.shape
        return maps_sub_20Q_train + random_noise, maps_sub_20U_train + random_noise
    
    else:
        return maps_sub_20Q_train, maps_sub_20U_train


def cl_anafast(map_QU, lmax):
    '''
    Return the full-sky power spetra, except monopole and dipole
    '''

    map_I = np.ones_like(map_QU[0])
    maps = np.row_stack((map_I, map_QU))
    cls_all = hp.anafast(maps, lmax = lmax)
    ells = np.arange(lmax+1)

    return ells[2:], cls_all[:, 2:]

def cl_nmt(nside, msk_apo, map_QU, lmax, nlbins, dl = False, w22_file = 'w22_2048_80_sky.fits', verbose = True):
    '''
    nside:
    msk_apo: apodized mask
    nlbins: ell-number in each bin
    '''
    
    binning = nmt.NmtBin(nside=nside, nlb=nlbins, lmax=lmax, is_Dell=dl)
    f2 = nmt.NmtField(msk_apo, [map_QU[0], map_QU[1]], purify_b=True)

    w22 = nmt.NmtWorkspace()
    try:
        w22.read_from(w22_file)
        if verbose: print('weights loaded from %s' % w22_file)
    except:
        w22.compute_coupling_matrix(f2, f2, binning)
        w22.write_to(w22_file)
        print('weights writing to disk')

    cl22 = nmt.compute_full_master(f2, f2, binning, workspace = w22)
    
    return binning.get_effective_ells(), cl22
    
def plot_spectra(cls_all, names, save_dir, lim = [1e-9, 1e3]):

    '''
    cls_80p_80amin = {'ells':ell_80p_80amin, 'spectra':cl_80p_80amin, 'color':'r-', 'label':'80amin'}
    cls_80p_12amin = {'ells':ell_80p_12amin, 'spectra':cl_80p_12amin, 'color':'g-', 'label':'12amin'}
    '''
    names = ['EE', 'BB']
    fig, axes = plt.subplots(1,2, figsize = (16, 5))

    for i in range(len(cls_all)):
        ells = cls_all[i]['ells']; cl = cls_all[i]['spectra']; color = cls_all[i]['color']; label = cls_all[i]['label']
        axes[0].loglog(ells, abs(cl[0]), color, label = label)
        axes[1].loglog(ells, abs(cl[3]), color)

    line1 = Line2D([],[],linestyle='-', color='r')
    line2 = Line2D([],[],linestyle='-.', color='r')
    line3 = Line2D([],[],linestyle='--', color='r')
    axes[0].legend(loc = 'lower left')
    axes[1].legend([line1, line2, line3],['full sky', '80% sky', '40% sky'], loc = 'lower left')

    for j in range(2):
        axes[j].set_ylim(lim[0], lim[1])
        axes[j].set_title('%s'%names[j])
        if j == 0:
            axes[j].set_ylabel(r'C$\ell$')
        axes[j].set_xlabel(r'$\ell$')

    if save_dir:
        plt.savefig(save_dir, format = 'pdf')  
        
    return fig
