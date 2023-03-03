# Functions that are specific to the NIRSpec instrument on JWST
from astropy.io import fits
from astropy.table import Table
from jwst_templates import spec, continuum
import numpy as np
import os

def load_dispersion(grating) : #grating as in 'prism', 'g140m', 'g235h'
    refdir = os.path.join(os.path.dirname(__file__), 'Reference_files/')
    filename = 'jwst_nirspec_' + grating.lower() + '_disp.fits'
    print("DEBUG", filename)
    RR, Rheader = fits.getdata(refdir + filename, header=True)   # old way
    df_R = Table.read(refdir + filename).to_pandas()             # ah, pandas  
    # WAVELENGTH will be wavelength in micron, DLDS will be dispersion in microns per pixel; R is dimensionless spectral resolution
    return(df_R, Rheader)

def load_filter_throughput(filtname):
    # Get the transmission curve for a NIRSpec filter
    refdir = os.path.join(os.path.dirname(__file__), 'Reference_files/')
    filename = 'jwst_nirspec_' + filtname.lower() + '_trans.fits'
    print("DEBUG", filename)
    throughput, header = fits.getdata(refdir + filename, header=True)   # old way
    df_throughput  = Table.read(refdir + filename).to_pandas()          # ah, pandas
    return(df_throughput, header)

def load_filter_and_grating(grating, filtname):  # Wrapper, what you prob want
    # Return a pandas dataframe with resolution, dispersion, and resampled throughput for a given nirspec filter, grating combo
    df_R, head_R = load_dispersion(grating)
    df_throughput, head_throughput = load_filter_throughput(filtname)
    # Put the throughput on the same wavelength grid as the spectral resolution
    df_R['throughput'] = spec.rebin_spec_new(df_throughput.WAVELENGTH, df_throughput.THROUGHPUT, df_R.WAVELENGTH, return_masked=False)
    return(df_R)
   
def calc_avg_specR(grating, filtname, op=np.nanmean):
    df_R = load_filter_and_grating(grating, filtname)
    # Calculate the weighted average spectral resolution and dispersion for a given NIRSpec filter, grating combo
    weighted_R    = (df_R.R    * df_R.throughput).sum() / df_R.throughput.sum()  # Weight by throughput, paging Bevington
    weighted_disp = (df_R.DLDS * df_R.throughput).sum() / df_R.throughput.sum()   
    return(weighted_R, weighted_disp)
    

# Need to add error trapping if the filter name or grating name is not recognized 
