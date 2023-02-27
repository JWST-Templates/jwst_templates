# Spectroscopy helper functions for JWST ERS program TEMPLATES

import numpy as np

from scipy.optimize import curve_fit
from specutils.manipulation import extract_region # to get sub-regions

from specutils.spectra import Spectrum1D
from specutils.fitting.continuum import fit_continuum
from specutils import SpectralRegion

from astropy.io import fits
import astropy.units as u
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty

import os


def load_nirspec_dispersion(grating) : #grating as in 'prism', 'g140m', 'g235h' 
    ndir = '/Users/jrrigby1/Python/TEMPLATES/jwst_templates/Reference_files/' # this is hardcoded.  Help having it look at the module?
    Rfilename = 'jwst_nirspec_' + grating.lower() + '_disp.fits'
    RR, Rheader = fits.getdata(ndir + Rfilename, header=True)
    return(RR, Rheader)


# some specific stuff for JWST IFU spectra:

def convert_jwst_to_cgs(spaxel, pixar_sr):
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1
    
    spaxflux = spaxel.flux
    spaxwave = spaxel.spectral_axis
    spaxerr = spaxel.uncertainty

    flux_jy = (spaxflux * pixar_sr).to(u.Jy)
    flux_cgs = flux_jy.to(cgs, equivalencies=u.spectral_density(spaxwave))
    err_jy = (spaxerr.array * u.MJy * u.sr**-1 * pixar_sr).to(u.Jy)
    err_cgs = err_jy.to(cgs, equivalencies=u.spectral_density(spaxwave))
    err_cgs = StdDevUncertainty(err_cgs)

    spaxel_cgs = Spectrum1D(flux=flux_cgs, spectral_axis=spaxwave, uncertainty=err_cgs)
    return spaxel_cgs


def extract1D_cube_mask(cube, mask, pixar_sr, zsource=None, errmode=None, operation=np.nansum ):
    '''
    Extract a 1D spectrum from an IFU cube given a pre-made mask
    Converts output to CGS units
    '''
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1
    
    cubewl = cube.spectral_axis
    cubeflux = cube.flux
    cuberr = cube.uncertainty
    
    cubeflux_jy = (cubeflux * pixar_sr).to(u.Jy)
    cubeflux_cgs = cubeflux_jy.to(cgs, equivalencies=u.spectral_density(cubewl))
    cuberr_jy = (cuberr.array * u.MJy * u.sr**-1 * pixar_sr).to(u.Jy)
    cuberr_cgs = cuberr_jy.to(cgs, equivalencies=u.spectral_density(cubewl))
    #cuberr_cgs = StdDevUncertainty(cuberr_cgs)
    flux0 = cubeflux_cgs
    err0 = cuberr_cgs
    
    if mask.shape != flux0[:,:,0].shape:
        mask = mask.T

    flux1 = []
    err1 = []
    for i in range(0,cubewl.shape[0]):
        flux_i = operation(flux0[:,:,i].value * mask)
        flux1.append(flux_i)
    flux1 = np.array(flux1) * cgs

    if errmode == 'bullshit':
        err1 = np.ones_like(flux1)
    else:
        for i in range(0,cubewl.shape[0]):
            err_i = operation(err0[:,:,i].value * mask)
            err1.append(err_i)
        err1 = np.array(err1)
    err_out = StdDevUncertainty(err1)

    if zsource:
        spec_out = Spectrum1D(flux=flux1, spectral_axis=cubewl/(1+zsource), uncertainty=err_out)
    else:
        spec_out = Spectrum1D(flux=flux1, spectral_axis=cubewl, uncertainty=err_out)

    return(spec_out)




# CURVE FITTING below here. From the several packages that fit curves, we use curve_fit because it reports uncertainties.

def gaussian(x, amp, mu, sigma):
    exponential = -1 * (x - mu)**2 / (2 * sigma**2)
    return amp * np.exp(exponential)

def multigauss(x, paramdict):
    amps = paramdict["amplitude"]
    mus = paramdict["mean"]
    sigs = paramdict["sigma"]
    many_g = 0
    for i in range(len(amps)):
        many_g += gaussian(x, amps[i], mus[i], sigs[i])
    return many_g

def twogauss(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    return gaussian(x, amp1, mu1, sigma1) + gaussian(x, amp2, mu2, sigma2)

def threegauss(x, amp1, mu1, sigma1, amp2, mu2, sigma2, amp3, mu3, sigma3):
    return gaussian(x, amp1, mu1, sigma1) + gaussian(x, amp2, mu2, sigma2) + gaussian(x, amp3, mu3, sigma3)


def cont_sub_curvefit(spectrum,  cont_region, line_region, line_params, verbose=False, scale=1e20):
    '''
    use scipy.optimize.curve_fit to fit a single emission line (generalize later to multi lines)
    this will also give an uncertainty on each fit param, and thus on the flux, EW, etc. 
    assumes spectrum is a specutils.Spectrum1D
    assumes line_region is a specutils.SpectralRegion
    scale exists to set numbers closer to 1 so curve_fit doesn't get mad at me. it should be ~1/flux
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
    print(wlunit)

    # now we fit!
    popt, pcov = curve_fit(gaussian, fit_wl, fit_flux, p0=line_params, sigma=fit_fluxerr, absolute_sigma=False)
    perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter

    fit_amp, fit_mu, fit_sigma = popt
    fit_amp /= scale # dividing by scale factor to make outputs make sense. 
    d_amp, d_mu, d_sigma = perr 
    d_amp /= scale # thanks code gremlins, look what you make us do!

    if wlunit == 'um':
        fit_sigma *= 1e4  #convert sigma from micron to angstrom so output unit makes sense
    flux_out = np.sqrt(2*np.pi) * fit_amp * np.abs(fit_sigma) 
    fluxerr_out = np.sqrt((d_amp/fit_amp)**2 + (d_sigma/np.abs(fit_sigma))**2) * flux_out

    if verbose:
        return flux_out, fluxerr_out, popt, pcov, contsub_spec
    else:
        return flux_out, fluxerr_out



def cont_sub_2curvefit(spectrum,  cont_region, line_region, line_params, verbose=False, scale=1e20):
    '''
    use scipy.optimize.curve_fit to fit a sdouble gaussian to two blended/nearby emission lines
    this will also give an uncertainty on each fit param, and thus on the flux, EW, etc. 
    assume line_region is a specutils SpectralRegion, and is continuous
    scale exists to set numbers closer to 1 so curve_fit doesn't get mad at me. it should be ~1/flux
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
    #print(min(fit_wl), max(fit_wl))
    # multiplying by scale factor to make fit function gremlins happy
    fit_flux = fit_spectrum.flux.value * scale
    fit_fluxerr = fit_spectrum.uncertainty.array * scale

    # now we fit!
    popt, pcov = curve_fit(twogauss, fit_wl, fit_flux, p0=line_params, sigma=fit_fluxerr, absolute_sigma=False)
    perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter

    fit_amp1, fit_mu1, fit_sigma1, fit_amp2, fit_mu2, fit_sigma2 = popt
    fit_amp1 /= scale # dividing by scale factor to make outputs make sense. 
    fit_amp2 /= scale
    d_amp1, d_mu1, d_sigma1, d_amp2, d_mu2, d_sigma2 = perr 
    d_amp1 /= scale # thanks code gremlins, look what you make us do!
    d_amp2 /= scale
    if wlunit == 'um':
        fit_sigma1 *= 1e4  #convert sigma from micron to angstrom so output unit makes sense
        fit_sigma2 *= 1e4
    flux_out1 = np.sqrt(2*np.pi) * fit_amp1 * np.abs(fit_sigma1) #convert sigma from micron to angstrom so output unit makes sense
    fluxerr_out1 = np.sqrt((d_amp1/fit_amp1)**2 + (d_sigma1/np.abs(fit_sigma1))**2) * flux_out1
    flux_out2 = np.sqrt(2*np.pi) * fit_amp2 * np.abs(fit_sigma2) #again for second line
    fluxerr_out2 = np.sqrt((d_amp2/fit_amp2)**2 + (d_sigma2/np.abs(fit_sigma2))**2) * flux_out2

    if verbose:
        return flux_out1, fluxerr_out1, flux_out2, fluxerr_out2, popt, pcov, contsub_spec
    else:
        return flux_out1, fluxerr_out1, flux_out2, fluxerr_out2


def cont_sub_3curvefit(spectrum,  cont_region, line_region, line_params, verbose=False, scale=1e20):
    '''
    use scipy.optimize.curve_fit to fit a sdouble gaussian to two blended/nearby emission lines
    this will also give an uncertainty on each fit param, and thus on the flux, EW, etc. 
    assume line_region is a specutils SpectralRegion, and is continuous
    scale exists to set numbers closer to 1 so curve_fit doesn't get mad at me. it should be ~1/flux
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
    #print(min(fit_wl), max(fit_wl))
    # multiplying by scale factor to make fit function gremlins happy
    fit_flux = fit_spectrum.flux.value * scale
    fit_fluxerr = fit_spectrum.uncertainty.array * scale

    # now we fit!
    popt, pcov = curve_fit(threegauss, fit_wl, fit_flux, p0=line_params, sigma=fit_fluxerr, absolute_sigma=False)
    perr = np.sqrt(np.diag(pcov)) # calculate 1-sigma uncertainties on each parameter

    fit_amp1, fit_mu1, fit_sigma1, fit_amp2, fit_mu2, fit_sigma2, fit_amp3, fit_mu3, fit_sigma3 = popt
    fit_amp1 /= scale # dividing by scale factor to make outputs make sense. 
    fit_amp2 /= scale
    fit_amp3 /= scale
    d_amp1, d_mu1, d_sigma1, d_amp2, d_mu2, d_sigma2, d_amp3, d_mu3, d_sigma3 = perr 
    d_amp1 /= scale # thanks code gremlins, look what you make us do!
    d_amp2 /= scale
    d_amp3 /= scale
    if wlunit == 'um':
        fit_sigma1 *= 1e4
        fit_sigma2 *= 1e4  #convert sigma from micron to angstrom so output unit makes sense
        fit_sigma3 *= 1e4
    flux_out1 = np.sqrt(2*np.pi) * fit_amp1 * np.abs(fit_sigma1) #convert sigma from micron to angstrom so output unit makes sense
    fluxerr_out1 = np.sqrt((d_amp1/fit_amp1)**2 + (d_sigma1/np.abs(fit_sigma1))**2) * flux_out1
    flux_out2 = np.sqrt(2*np.pi) * fit_amp2 * np.abs(fit_sigma2) #again for second line
    fluxerr_out2 = np.sqrt((d_amp2/fit_amp2)**2 + (d_sigma2/np.abs(fit_sigma2))**2) * flux_out2
    flux_out3 = np.sqrt(2*np.pi) * fit_amp3 * np.abs(fit_sigma3) #again for third line
    fluxerr_out3 = np.sqrt((d_amp3/fit_amp3)**2 + (d_sigma3/np.abs(fit_sigma3))**2) * flux_out3

    if verbose:
        return flux_out1, fluxerr_out1, flux_out2, fluxerr_out2, flux_out3, fluxerr_out3, popt, pcov, contsub_spec
    else:
        return flux_out1, fluxerr_out1, flux_out2, fluxerr_out2, flux_out3, fluxerr_out3


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





