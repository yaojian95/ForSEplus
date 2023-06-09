import healpy as hp
import pylab as pl
import astropy
from astropy import units as u
import collections
import reproject
import numpy as np
import astropy.io.fits as fits

import argparse
import time
import warnings
warnings.filterwarnings("ignore")


def set_header(ra,dec, size_patch ,Npix=128 ):
    """
    Set the header of a fits file. This is useful for the reprojection.
    We assume to reproject square patches.

    **Parameters**

    - `ra` : {float}
    longitudinal coordinate of the patch we want to reproject
    - `dec` : {float}
    latitudinal coordinate of the patch we want to reproject
    - `size_patch` :{astropy. quantity [u.deg]}
    size of the  patch  side  in degrees
    - `Npix` :{int}
    number of pixel on one side of the patch


    **Returns**

    - Fits header related to the coordinates of the centroid
    """

    hdr = fits.Header()
    hdr.set('SIMPLE' , 'T')
    hdr.set('BITPIX' , -32)
    hdr.set('NAXIS'  ,  2)
    hdr.set('NAXIS1' ,  Npix)
    hdr.set('NAXIS2' ,  Npix )
    hdr.set('CRVAL1' ,  ra)
    hdr.set('CRVAL2' ,  dec)
    hdr.set('CRPIX1' ,  Npix/2. +.5)
    hdr.set('CRPIX2' ,  Npix/2. +.5 )
    hdr.set('CD1_1'  , size_patch )
    hdr.set('CD2_2'  , -size_patch )
    hdr.set('CD2_1'  ,  0.0000000)
    hdr.set('CD1_2'  , -0.0000000)
    hdr.set('CTYPE1'  , 'RA---ZEA')
    hdr.set('CTYPE2'  , 'DEC--ZEA')
    hdr.set('CUNIT1'  , 'deg')
    hdr.set('CUNIT2'  , 'deg')
    hdr.set('COORDSYS','icrs')
    return hdr

def h2f(hmap,target_header,coord_in='C'):
    """
    project healpix map  -> flatsky

    Interface to the `reproject.reproject_from_healpix`

    **Parameters**

    - `hmap`: {array}
    healpix map
    - `target_header`: {FITS header}
    the output of `set_header`
    - `coord_in` :{string}
    coordinate frame of the input map, 'C' for Celestial, 'E' for ecliptical,
    `G` for Galactic

    **Returns**
    - The reprojected map in a flat pixelization
    """

    pr,footprint = reproject.reproject_from_healpix(
    (hmap, coord_in), target_header, shape_out=(500,500),
    order='nearest-neighbor', nested=False)
    return pr

def f2h(flat,target_header,nside,coord_in='C'):
    """
    #project flatsky->healpix
    Interface to the `reproject.reproject_to_healpix`

        **Parameters**

        - `flat`: {2D array}
        map in flat coordinate
        - `target_header`: {FITS header}
        the output of `set_header`
        - `coord_in` :{string}
        coordinate frame for  the output map, 'C' for Celestial, 'E' for ecliptical,
        `G` for Galactic
        - `nside`:{int}
        the output healpix pixelization parameter

        **Returns**

        - `pr`:The reprojected map in a healpix  pixelization
        - `footprint`:  the footprint of the repojected map as a binary healpix mask
    """

    pr,footprint = reproject.reproject_to_healpix(
    (flat, target_header),coord_system_out='C', nside=nside ,
    order='nearest-neighbor', nested=False)
    return pr, footprint


def get_lonlat( size_patch ,  overlap ):
    """
    Divide the whole sphere into patches with sizes given by `size_patch`(in degrees) and with a certain overlap
    given in units of degrees. Notice that the number of patches overlapping increases at the poles and decreases at
    the equator.


    **Parameters**

    - `size_patch` :{astropy. quantity [u.deg]}
    - `overlap` :{astropy. quantity [u.deg]}

    **Returns**

    - `lon`:{np.array}
    the longitudinal coordinates of the patches
    -`lat`:{np.array}
    the latitudinal coordinate of the patches

    """

    Nlon = np.int_( np.ceil(360.*u.deg / (size_patch  -overlap  )  ).value  )
    Nlat =np.int_  (np.ceil( 180. *u.deg/( size_patch-overlap )  ).value )    +1

    rLon =(360.*u.deg % (size_patch  -overlap  )).value; rLat= (180. *u.deg%( size_patch-overlap ) ) .value

    offset_lon = 0
    offset_lat= -90
    lat_array= np.zeros(np.int_(Nlat) )
    lon_array= np.zeros(np.int_(Nlon) )
    lat_array[:Nlat] = [ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat) ]
    lon_array[:Nlon ] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon ) ]

    if rLon==0 and rLat==0:
        lat_array[:Nlat] = [ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat) ]
        lon_array[:Nlon ] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon ) ]
    else :
        lat_array[:Nlat-1] = [ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat-1) ]
        lon_array[:Nlon -1] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon-1 ) ]
        lat_array[Nlat-1] =  lat_array[-2 ]+ rLat
        lon_array[Nlon-1] =  lon_array[-2 ]+ rLon

    lon , lat  =pl.meshgrid(lon_array,lat_array)
    return   lon.ravel(),lat.ravel()


def get_lonlat_adaptive  ( size_patch ,  overlap   ):
    """
    Divide the whole sphere into patches with sizes given by `size_patch`(in degrees) and with a certain overlap
    given in units of degrees. To avoid the fact that the number of patches overlapping increases at the poles and decreases at
    the equator, we implemented an adaptive division of the sphere having less overlapping patches  at the poles.


    **Parameters**

    - `size_patch` :{astropy. quantity [u.deg]}
    - `overlap` :{astropy. quantity [u.deg]}


    **Returns**

    - `lon`:{np.array}
    the longitudinal coordinates of the patches
    - `lat`:{np.array}
    the latitudinal coordinate of the patches

    """
    Nlon = np.int_( np.ceil(360.*u.deg / (size_patch  -overlap  )  ).value  )
    Nlat =np.int_  (np.ceil( 180. *u.deg/( size_patch-overlap )  ).value )
    if Nlat %2 ==0 :
        Nlat+= 1
    offset_lon = 0
    offset_lat= -90
    lat_array= np.zeros(np.int_(Nlat) )
    lon_array= np.zeros(np.int_(Nlon) )

    lat_array[:Nlat//2   ] =[ offset_lat + ((size_patch ).value -overlap.value) *i for i in range(  Nlat//2  )  ]
    lat_array[Nlat//2+1 : ] =[ -offset_lat -  ((size_patch ).value -overlap.value) *i for i in range(  Nlat//2 ) ] [::-1]
    lat_array[Nlat//2] =  0
    lon_array[:Nlon ] = [offset_lon + ((size_patch ).value -overlap.value )*i for i in range( Nlon ) ]
    Nloneff = np.int_( np.cos(np.radians(lat_array))*Nlon )
    Nloneff[0]=5; Nloneff[-1] =5
    jumps = (np.int_(np.ceil(Nlon/Nloneff ) -1)  )
    jumps[1] -=1
    jumps[-2] -=1
    jumps [Nlat//2] =1
    lon , lat  =pl.meshgrid(lon_array,lat_array)
    lonj=[]
    latj =[]
    for kk in range(Nlat ):
        lonj.append(lon[kk,::jumps[kk] ])
        latj.append(lat[kk,::jumps[kk]] )
    lonj=np.concatenate(lonj); latj=np.concatenate(latj)
    return lonj, latj


def make_mosaic_from_healpix( hpxmap , Npix, pixel_size , overlap, adaptive=False  ):
    """
    Make a mosaic of patches from a healpix map.

    **Parameters**

    - `hpxmap`:{array}
    the map to project to flat
    - `Npix` : {int}
    the number of pixels in the flat image side
    - `pixel_size`: {astropy.unit.arcmin }
    the pixel size in the flat projection
    - `overlap`:  {float  }
    the overlap in degrees between two neighbour patches; [=5 *u.deg]
    - `adaptive`: {bool}
    if `True` performs the projection in an adaptive way, to reduce overlaps at the poles.

    **Returns**

    - `patches`: {array}
    the patches in the format of an array of 2D patches, `shape= (Npatches, Npix,Npix )`
    - `lon` :{array}
    the longitudinal coordinate of all the projected patches
    - `lat`:{array}
    the latitudinal coordinate of all the projected patches

    """


    patches = []
    size_patch = pixel_size *Npix
    if adaptive:
        lon,lat = get_lonlat_adaptive(size_patch ,   overlap=overlap )

    else:
        lon,lat = get_lonlat(size_patch ,overlap= overlap )

    for  phi,theta in zip (lon , lat   ):
        header = set_header(phi, theta, pixel_size.value , Npix)
        patches.append(h2f(hpxmap , header))

    patches = np.array( patches )
    return patches, lon , lat



def reproject2fullsky ( tiles, lon, lat ,
                        nside_out, pixel_size, Npix, apodization_file=None ,
                       verbose=False, comm=None   ):
    """
    Project  square  patches into  a healpix map. When reprojected back to healpix,
    to ensure continuity between neighboring patches  an apodization window can be applied
    to the square patches to smooth a bit the edge features .

    **Parameters**

    - `patches`: {array}
    the patches in the format of an array of 2D patches, `shape= (Npatches, Npix,Npix )`
    - `lon` :{array}
    the longitudinal coordinate of all the projected patches
    - `lat`:{array}
    the latitudinal coordinate of all the projected patches
    - `nside_out`:{int}
    the map to project to flat
    - `Npix` : {int}
    the number of pixels in the flat image side
    - `pixel_size`: {astropy.unit.arcmin }
    the pixel size in the flat projection
    - `apodization_file`:  {string}
    path to the apodization window function `.npy` file, the dimension must be `Npix x Npix`.
    - `verbose`: {bool}
    to include verbosity in the run
    - `comm` : {MPI communicator}
    the MPI communicator needed only for debugging issues


    **Returns**

    - `newmap`:{array}
    HEALPIX map reprojected
    - `weightsmap`:{array}
    the healpix map accounting for how many times each location in the sky have been
    projected into a healpix pixel.

    .. note::

    the final healpix map needs to be  divided by the weights which are variable for each location in the sky,
    as  :

        `reprojected_healpix_map = newmap/weightsmap`

    """
    if comm is None :
        rank =0
    else :
        rank =comm.Get_rank()

    newmap =pl.zeros(hp.nside2npix((nside_out)) )
    weightsmap = pl.zeros_like(newmap )
    flatmap = pl.ones ((Npix,Npix))
    sizedeg= pixel_size.to(u.deg)
    development= 0
    s=time.time()
    try:
        apoflat = pl.load(apodization_file )
    except TypeError:
        apoflat=flatmap

    for p, phi, theta  in zip ( tiles , lon  ,lat  )  :
        header = set_header(phi, theta , sizedeg.value ,Npix  )

        tmpmap,fp= f2h (p*apoflat ,header ,nside_out  )
        tmpmap [ pl.ma.masked_invalid(tmpmap).mask  ]=0

        newmap+=tmpmap
        weightsmap +=fp

        development +=1


    e=time.time()
    if  rank ==0 : print(f"   {len(tiles)} projected  in {e-s} sec" )

    return newmap, weightsmap


def make_patches_from_healpix(
        Npatches, m_hres, m_lres, Npix, patch_dim, lat_lim=None, seed=None, mask=None):
    high_res_patches = []
    low_res_patches = []
    reso_amin = patch_dim*60./Npix
    sizepatch = reso_amin/60.
    if seed:
        np.random.seed(seed)
    if np.any(mask)==None:
        mask_hp = m_hres*0.+1
    else:
        mask_hp = mask
    for N in range(Npatches):
        if lat_lim:
            lat = np.random.uniform(-lat_lim,lat_lim)
        else:
            lat = np.random.uniform(-90,90)
        lon = np.random.uniform(0,360)
        header = set_header(lon, lat, sizepatch, Npix)
        mask_patch = h2f(mask_hp, header)
        if len(np.where(mask_patch>0)[0])/(Npix*Npix)>0.9:
            if len(m_hres)>3:
                high_res_patches.append(h2f(m_hres, header))
                low_res_patches.append(h2f(m_lres, header))
            else:
                high_res_patch_TQU = np.zeros((len(m_hres), Npix, Npix))
                low_res_patch_TQU = np.zeros((len(m_lres), Npix, Npix))
                for i in range(len(m_hres)):
                    high_res_patch_TQU[i] = h2f(m_hres[i], header)
                    low_res_patch_TQU[i] = h2f(m_lres[i], header)
                high_res_patches.append(high_res_patch_TQU)
                low_res_patches.append(low_res_patch_TQU)
    patches = np.array([high_res_patches, low_res_patches])
    return patches
