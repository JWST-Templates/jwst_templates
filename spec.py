# Spectroscopy helper functions for JWST ERS program TEMPLATES

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from specutils.manipulation import extract_region # to get sub-regions

from specutils.spectra import Spectrum1D
from specutils.fitting.continuum import fit_continuum
from specutils import SpectralRegion

from astropy.io import fits
import astropy.units as u
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty

import os

def rebin_spec_new(wave, specin, new_wave, kind='linear', fill=np.nan, return_masked=False):
    # By default, this does linear interpolation (kind='linear').
    # Rebin spectra (wave, specin) to new wavelength aray new_wave.  Fill values are nan.  If return_masked, then mask the nans
    # this is copied from janerigby's github, jrr/spec.py
    f = interp1d(wave, specin, kind=kind, bounds_error=False, fill_value=fill)  # With these settings, writes NaN to extrapolated regions
    new_spec = f(new_wave)
    if return_masked :  return(np.ma.masked_array(data,np.isnan(new_spec)))
    else :              return(new_spec)



def get_matrix_coords(dims):
    '''
    Given an input 3D cube dimensions, it will convert the 
    x,y spaxel coordinates into a list.
    
    Input can be written as:
    >> get_matrix_coords(cube.shape) # easiest method
    >> get_matrix_coords((1000,50,50)) # if manually inputting dims

    '''
    x0,y0 = np.arange(0,dims[2]),np.arange(0,dims[1])
    g = np.meshgrid(x0,y0)
    coords = list(zip(*(c.flat for c in g))) # coords we'll walk through
    return coords

        
        
# some specific stuff for JWST IFU spectra:


def convert_jwst_to_cgs(spaxel, pixar_sr):
    '''
    INPUTS:
    >> spaxel  --------  a pandas dataframe set up with columns
                         "wave", "flux", "ferr"; assumes wave is 
                         in microns and flux, ferr are in MJy
    OUTPUTS:
    >> spaxel ---------  the same pandas dataframe but in cgs
    '''
    # converting from MJy/sr to MJy
    spaxel['flam'] *= pixar_sr # MJy/sr --> MJy
    spaxel['flamerr'] *= pixar_sr # MJy/sr --> MJy
    
    # converting spectrum flux to cgs units
    spaxel['flam'] *= 1e6 # MJy --> Jy
    spaxel['flam'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spaxel['flam'] *= 2.998e18 / (spaxel.wave.values*1e4)**2 # fnu --> flam (erg/s/cm^2/AA)
    
    # converting spectrum error to cgs units
    spaxel['flamerr'] *= 1e6 # MJy --> Jy
    spaxel['flamerr'] *= 1e-23 # Jy --> erg/s/cm/Hz (fnu)
    spaxel['flamerr'] *= 2.998e18 / (spaxel.wave.values*1e4)**2 # fnu --> flam (erg/s/cm^2/AA)
    
    return spaxel.copy()




def integrate1D_mask(cubefile, maskfile, zsource=None, operation=np.nansum):
    '''
    Creates an integrated 1D spectrum from an IFU cube file given a pre-made mask.
    Inputs are fits files, one for the IFU cube (cubefile) and one for the mask (maskfile).
    If a redshift (zsource) is given, a rest-frame wavelength will be calculated.
    Output will be a pandas dataframe with wavelength, flam (erg/s/cm^2/A), flamerr (erg/s/cm^2/A).
    If zsource is given, rest_wave column will be added to dataframe
    '''
    # reading in IFU cube and header info
    hdu = fits.open(cubefile)
    header = hdu[1].header
    dat = hdu[1].data  # in MJy/sr
    err = hdu[2].data  # error, same units as dat
    
    # using header info
    pixar_sr = header['PIXAR_SR'] # pixel area, in sr
    wl = np.arange(header['CRVAL3'],  # wavelength in micron
            header['CRVAL3']+(header['CDELT3']*len(dat)), 
            header['CDELT3'])
    
    # reading in the mask of the galaxy (0=not galaxy, 1=galaxy)
    mask = fits.open(maskfile)[0].data
    if mask.shape != dat[0].shape:
        mask = mask.T
        
    # broadcasting mask to IFU cube
    longmask = np.broadcast_to(mask, dat.shape)
    
    # using chosen operation on flux density and associated error
    flux = operation(dat * longmask, axis=(1,2))
    error = np.sqrt(operation(err**2 * longmask, axis=(1,2)))
    
    # making dataframe of final spectrum
    df = pd.DataFrame({'wave':wl, 'flam':flux, 'flamerr':error})
    df_cgs = convert_jwst_to_cgs(df, pixar_sr)
    
    # if source has a redshift, will add rest wavelength column
    if zsource:
        wl_rest = wl / (1+zsource)
        df_cgs["rest_wave"] = wl_rest
        
    return df_cgs



# CURVE FITTING below here. From the several packages that fit curves, we use curve_fit because it reports uncertainties.

def gaussian(x, amp, mu, sigma):
    exponential = -1 * (x - mu)**2 / (2 * sigma**2)
    return amp * np.exp(exponential)


def multigauss2(x, *args):
    ngauss = int(len(args) / 3)
    total = 0
    for n in range(ngauss):
        i = 3*n
        j = 3*n + 1
        k = 3*n + 2
        total += gaussian(x, args[i], args[j], args[k])
    return total    


def cont_sub_curvefit(spectrum,  cont_region, line_region, line_param_dict, verbose=False, scale=1e20):
    '''
    use scipy.optimize.curve_fit to fit N gaussians to N emission lines
    this will also give an uncertainty on each fit param, and thus on the flux, EW, etc. 
    assume spectrum is a specutile Spectrum1D object
    assume cont_region, line_region is a specutils SpectralRegion
    assume line_param_dict is a dictionary with keys "amplitude", "wavelength", and "width"
    corresponding to gaussian amplitude, central wavelength (gauss mean), and linewidth (gauss sigma)
    scale exists to set numbers closer to 1 so curve_fit doesn't get mad at me. it should be ~1/flux
    returns list of line fluxes, flux errors
    if verbose, also returns curve_fit outputs (popt, pcov) and continuum-subtracted spectrum used in fitting
    '''
    # first continuum subtract, same as before w/ specutils
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

    spaxwave = spectrum.spectral_axis

    fitted_cont = fit_continuum(spectrum, window=cont_region)
    y_cont = fitted_cont(spaxwave)

    contsub_spec = spectrum - y_cont

    # now we extract the line region
    line_min, line_max = line_region.lower.value, line_region.upper.value

    fit_spectrum = extract_region(contsub_spec, line_region, return_single_spectrum=True)

    if max(fit_spectrum.spectral_axis.value) < 1:
        fit_wl = fit_spectrum.spectral_axis.value * 1e4 # convert to angstrom to appease fitting gremlins
        wlunit = 'AA'
    else:
        fit_wl = fit_spectrum.spectral_axis.value # don't convert if microns are greater than 1
        wlunit = 'um'

    # multiplying by scale factor to make fit function gremlins happy
    fit_flux = fit_spectrum.flux.value * scale
    fit_fluxerr = fit_spectrum.uncertainty.array * scale

    # Unpack line param dict into single line param array
    amps = line_param_dict["amplitude"]
    means = line_param_dict["wavelength"]
    sigs = line_param_dict["width"]
    # for convenience:
    ngauss = len(amps)
    line_params = np.zeros(3*ngauss)
    for i in range(ngauss):
        line_params[3*i] = amps[i]
        line_params[3*i + 1] = means[i]
        line_params[3*i + 2] = sigs[i]


    # now we fit!
    popt, pcov = curve_fit(multigauss2, fit_wl, fit_flux, p0=line_params, sigma=fit_fluxerr, absolute_sigma=False)
    perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter
    
    fluxlist = []
    fluxerrlist = []
    for n in range(ngauss):
        i, j, k = 3*n, 3*n+1, 3*n+2

        fit_amp, fit_mu, fit_sigma = popt[i], popt[j], popt[k]
        fit_amp /= scale # dividing by scale factor to make outputs make sense. 
        d_amp, d_mu, d_sigma = perr[i], perr[j], perr[k]
        d_amp /= scale # thanks code gremlins, look what you make us do!

        if wlunit == 'um':
            fit_sigma *= 1e4  #convert sigma from micron to angstrom so output unit makes sense
        flux = np.sqrt(2*np.pi) * fit_amp * np.abs(fit_sigma) 
        fluxlist.append(flux)
        fluxerr = np.sqrt((d_amp/fit_amp)**2 + (d_sigma/np.abs(fit_sigma))**2) * flux
        fluxerrlist.append(fluxerr)


    if verbose:
        return fluxlist, fluxerrlist, popt, pcov, contsub_spec
    else:
        return fluxlist, fluxerrlist




# I was going to write something using scipy least_squares, 
# but scipy curve_fit is built on least_squares, so that's just reinventing the wheel?


########################################
# And for my next trick, set some upper limits!
# woohoo!
########################################

# first, lets define if a line exists:

def is_line(spectrum, line_region, cont_region, n=3.):
    '''
    Check to see if a line is present
    Takes an expected line region, and a continuum region
    First does continuum substraction
    Next takes standard deviation of continuum region
    If the peak of the line region is > n*sigma above zero, we say there is a line, and we party!
    '''
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

    spaxwave = spectrum.spectral_axis

    fitted_cont = fit_continuum(spectrum, window=cont_region)
    y_cont = fitted_cont(spaxwave)

    contsub_spec = spectrum - y_cont
    # maybe I should put all this continuum subtraction in a seperate function at some point? 
    # could be better than redoing it every time...

    # okay, take the std deviation of the continuum region!
    noise_spec = extract_region(contsub_spec, cont_region, return_single_spectrum=True)
    noiseflux = noise_spec.flux.value # should be a single array of flux values
    noise_sigma = np.std(noiseflux)

    # now we extract the line region
    line_min, line_max = line_region.lower.value, line_region.upper.value

    line_spec = extract_region(contsub_spec, line_region, return_single_spectrum=True)
    line_peak = max(line_spec.flux.value)

    # now, the decision! Will we take our talents to south beach?!?
    if line_peak > n * noise_sigma: return True
    else: return False


def flux_upper_limit(spectrum, line_region, cont_region, n=3, width=0.00015, verbose=False):
    '''
    Check to see if a line is present, then if not set an upper limit on flux
    Takes an expected line region, and a continuum region
    First does continuum substraction
    Next takes standard deviation of continuum region
    If the peak of the line region is > n*sigma above zero, we say there is a line, and we party!
    If the peak is less than n*sigma, we set an upper limit
    We assume a Gaussian line profile with amplitude n*sigma, and width set by "width" parameter (0.0015 micron default)
    that flux is our upper limit
    Should probably test this - inject a 3-sigma peak and see where if it can be recovered/fit
    that's a later Brian problem. Later Brian is so helpful. 
    '''
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

    spaxwave = spectrum.spectral_axis

    fitted_cont = fit_continuum(spectrum, window=cont_region)
    y_cont = fitted_cont(spaxwave)

    contsub_spec = spectrum - y_cont
    # maybe I should put all this continuum subtraction in a seperate function at some point? 
    # could be better than redoing it every time...

    # okay, take the std deviation of the continuum region!
    noise_spec = extract_region(contsub_spec, cont_region, return_single_spectrum=True)
    noiseflux = noise_spec.flux.value # should be a single array of flux values
    noise_sigma = np.std(noiseflux)

    # now we extract the line region
    line_min, line_max = line_region.lower.value, line_region.upper.value

    line_spec = extract_region(contsub_spec, line_region, return_single_spectrum=True)
    line_peak = max(line_spec.flux.value)

    # now, the decision! Will we take our talents to south beach?!?
    if line_peak > n * noise_sigma: 
        print("There's a line here! No upper limit needed! Yippee!")
        return -1
    else: 
        max_flux = np.sqrt(2*np.pi) * (n*noise_sigma) * (width * 1e4) # amp = n*sigma, convert width to Angstrom from Micron
        if verbose:
            return max_flux, noise_sigma, line_peak
        else:
            return max_flux





