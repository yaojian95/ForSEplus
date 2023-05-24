from reproject.utils import parse_input_data
from scipy.ndimage import map_coordinates
from .projection_tools import f2h, set_header, get_lonlat_adaptive

import healpy as hp
import numpy as np
from astropy import units as u

import time


class recom(object):
    
    '''
    reproject flat patches back to a healpix map
    '''
    
    def __init__(self, npix, pixelsize, overlap, nside, 
                 apodization_file, 
                 xy_inds_file, 
                 index_sphere_file, 
                 verbose = False):
        
        if npix == 320:
            print('12amin: Initializing the re-projection...')
        elif npix == 1280:
            print('3amin: Initializing the re-projection ...')
            
        self.verbose = verbose
        self.apoflat = np.load(apodization_file)
        self.apomap = hp.read_map(apodization_file.replace('.npy','.fits'))
        
        self.Npix= np.int_(npix)
        pixel_size = pixelsize*u.arcmin
        self.sizedeg= pixel_size.to(u.deg)

        overlap = overlap*u.deg
        nside_in = nside

        hpxsize  = hp.nside2resol(nside_in, arcmin=True )*u.arcmin
        self.nside_out = np.int_(nside_in)
        size_patch = pixel_size.to(u.deg)*self.Npix

        self.lon, self.lat =get_lonlat_adaptive(size_patch, overlap)
        
        self.xinds_yinds = np.load(xy_inds_file, allow_pickle=True)
        self.index_sphere = np.load(index_sphere_file, allow_pickle=True)
        
    def recompose_fast(self, patches174_file):

        if isinstance(patches174_file, str):
            patches174 = np.load(patches174_file)
        else:
            patches174 = patches174_file
        newmap =np.zeros(hp.nside2npix((self.nside_out)) )

        i  = 0
        s = time.time()
        for p, phi, theta  in zip (patches174, self.lon, self.lat)  :

            header = set_header(phi, theta , self.sizedeg.value, self.Npix )
            input_data = (p*self.apoflat ,header )
            array_in, wcs_in = parse_input_data(input_data, hdu_in=0)

            xinds, yinds= self.xinds_yinds[i]

            healpix_data = map_coordinates(array_in, [xinds, yinds], order=0, mode="constant", cval=np.nan) ####here
            footprint = (~np.isnan(healpix_data)).astype(float)
            healpix_data[np.ma.masked_invalid(healpix_data).mask] = 0

            newmap[self.index_sphere[i]]+=healpix_data

            i += 1
        e = time.time()
        if self.verbose:
            if self.Npix == 320:
                print('12amin: Complete the reprojection!')
            elif self.Npix == 1280:
                print('3amin: Complete the reprojection!')
            print('Reprojecting to full sky takes %.2f'%(e-s), 'seconds')
        return newmap/self.apomap