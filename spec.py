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





