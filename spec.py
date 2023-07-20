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

from . import continuum

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
                         in microns and flux, ferr are in MJy/sr
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




def integrate1D_mask(cubefile, maskfile, zsource=None, operation=np.nansum, maskext=0, debug_output=False):
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
    if len(wl) > len(dat):
        wl = wl[:-1] # deal with weird arange glitch adding an extra wl value

    # convert to CGS units:
    dat2 = dat * pixar_sr * 1e6 * 1e-23 # MJy/sr -> MJy -> Jy -> erg/s/cm2/Hz
    err2 = err * pixar_sr * 1e6 * 1e-23
    for i in range(len(wl)):
        dat2[i,:,:] *= 2.998e18 / (wl[i]*1e4)**2
        err2[i,:,:] *= 2.998e18 / (wl[i]*1e4)**2


    # reading in the mask of the galaxy (0=not galaxy, 1=galaxy)
    try: 
        temp = len(maskext) # triggers type error if maskext is an int
        for i in range(temp):
            if i == 0:
                mask = fits.open(maskfile)[maskext[i]].data
            else:
                mask_i = fits.open(maskfile)[maskext[i]].data 
                mask += mask_i
            mask[mask>1] = 1 # just in case anything gets added twice
    except TypeError: 
        mask = fits.open(maskfile)[maskext].data # ext 0 generally has full mask, others have smaller S/N layers from clipping
    if mask.shape != dat[0].shape:
        mask = mask.T
    
    #npix = len(mask[mask==1].flatten())
    # broadcasting mask to IFU cube
    longmask = np.broadcast_to(mask, dat.shape)
    dat3 = dat * longmask
    err3 = err * longmask
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
        
    if debug_output == True:
        return df_cgs, mask, dat, dat2, err2, dat3, err3
    else:
        return df_cgs



# CURVE FITTING below here. From the several packages that fit curves, we use curve_fit because it reports uncertainties.

def gaussian(x, amp, mu, sigma):
    exponential = -1 * (x - mu)**2 / (2 * sigma**2)
    return amp * np.exp(exponential)


def multigauss(x, *args, fixed=None):
    # set fixed as a list of tuples (index, value) that you wnat fixed
    # get list of inds from list of tuples:
    if fixed is not None:
        fixinds = [f[0] for f in fixed]
    ngauss = int(len(args) / 3)
    total = 0
    for n in range(ngauss):
        i = 3*n
        j = 3*n + 1
        k = 3*n + 2
        if fixed is not None:
            if i in fixinds: ival = fixed[fixinds.index(i)][1]
            else: ival = args[i]
            if j in fixinds: jval = fixed[fixinds.index(j)][1]
            else: jval = args[j]
            if k in fixinds: kval = fixed[fixinds.index(k)][1]
            else: kval = args[k]
        else: 
            ival = args[i]
            jval = args[j]
            kval = args[k]
        total += gaussian(x, ival, jval, kval)
    return total  


def cont_sub_curvefit(spectrum,  line_region, line_param_dict, zz=None, obs_wl=True, verbose=False, scale=1e20, fixed=None):
    '''
    use scipy.optimize.curve_fit to fit N gaussians to N emission lines
    this will also give an uncertainty on each fit param, and thus on the flux, EW, etc. 
    assume spectrum is a pandas dataframe (e.g. output from integrate1D_mask or fit_autocont)
    assume line_region is array_like, (lower_bound, upper_bound)
    assume line_param_dict is a dictionary with keys "amplitude", "wavelength", and "width"
    corresponding to gaussian amplitude, central wavelength (gauss mean), and linewidth (gauss sigma)
    zz = redshift passed to fit_autocont
    scale exists to set numbers closer to 1 so curve_fit doesn't get mad at me. it should be ~1/flux
    returns list of line fluxes, flux errors
    if verbose, also returns curve_fit outputs (popt, pcov) and continuum-subtracted spectrum used in fitting
    fixed == parameters to be held constant in fitting. Given as a list of strings of parameter names 
    (e.g. ["width","wavelength"] to fix the wavelength and line width of each parameter)
    '''
    # first continuum subtract, same as before w/ specutils
    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1

    if obs_wl == True:
        wavekey = 'wave'
    else:
        wavekey='rest_wave'
    spaxwave = spectrum[wavekey]

    if "flam_autocont" in spectrum:
        y_cont = spectrum["flam_autocont"]
    else:
        print("No continuum fit found - performing continuum fit now!")
        if zz == None:
            print ("No redshift given! Skipping continuum fit.")
        else:
            contfit_spectrum = continuum.fit_autocont(spectrum, zz, boxcar=51, v2mask=300, colwave='wave',colf='flux', colcont='flam_autocont') # need to tweak boxcar, v2mask tobe physical
            y_cont = contfit_spectrum["flam_autocont"]


    #fitted_cont = fit_continuum(spectrum, window=cont_region)
    #y_cont = fitted_cont(spaxwave)

    #contsub_spec = spectrum - y_cont

    # now we extract the line region
    line_min, line_max = line_region[0], line_region[1]

    #fit_spectrum = extract_region(contsub_spec, line_region, return_single_spectrum=True)
    fit_spectrum = spectrum[(spectrum[wavekey] > line_min) & (spectrum[wavekey] < line_max)]
    

    #if max(fit_spectrum[wavekey]) < 1:
    fit_wl = fit_spectrum[wavekey] * 1e4 # convert to angstrom to appease fitting gremlins
    wlunit = 'AA'
    #else:
    #    fit_wl = fit_spectrum[wavekey] # don't convert if microns are greater than 1
    #    wlunit = 'um'
    #print(wlunit)
    # multiplying by scale factor to make fit function gremlins happy
    fit_flux = (fit_spectrum.flam - fit_spectrum.flam_autocont) * scale 
    fit_fluxerr = fit_spectrum.flamerr * scale

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


    # handle any possible fixed parameters
    if fixed is not None: 
        fix_tuples = [] 
        if "amplitude" in fixed: 
            for i in range(ngauss):
                tup = (3*i, amps[i])
                fix_tuples.append(tup)
        if "wavelength" in fixed: 
            for i in range(ngauss):
                tup = (3*i + 1, means[i])
                fix_tuples.append(tup)
        if "width" in fixed: 
            for i in range(ngauss):
                tup = (3*i + 2, sigs[i])
                fix_tuples.append(tup)
    else: fix_tuples = None
    # now we fit!
    fit_y = lambda x,*args: multigauss(x, *args, fixed=fix_tuples)
    popt, pcov = curve_fit(fit_y, fit_wl, fit_flux, p0=line_params, sigma=fit_fluxerr, absolute_sigma=False)
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
        print(fit_amp, fit_sigma)
        flux = np.sqrt(2*np.pi) * fit_amp * np.abs(fit_sigma) 
        fluxlist.append(flux)
        fluxerr = np.sqrt((d_amp/fit_amp)**2 + (d_sigma/np.abs(fit_sigma))**2) * flux
        fluxerrlist.append(fluxerr)


    if verbose:
        return fluxlist, fluxerrlist, popt, pcov#, contsub_spec
    else:
        return fluxlist, fluxerrlist




# I was going to write something using scipy least_squares, 
# but scipy curve_fit is built on least_squares, so that's just reinventing the wheel?


########################################
# And for my next trick, set some upper limits!
# woohoo!
########################################


def get_uplim(spec, linereg, contreg, nsigma=3, 
              wavekey='rest_wave', fluxkey='flam', contkey='flam_autocont'):
    '''
    Get upper limit for non-detected emission line
    spec: spectrum dataframe
    linereg: region where your (alleged) emission line resides - array-like w/ lower, upper wavelength bound
    contreg: nearby region without emission lines - array-like, either 2 numbers (lower,upper) or 4 (low1,up1,low2,up2)
        low1,up1 and low2,up2 would be on opposite sides of the emission line region
    nsigma: significance of detection - default = 3
    wavekey: keyword for wavelength in dataframe
    fluxkey: keyword for flux in dataframe
    contkey: keyword for continuum flux in dataframe
    STEPS:
    calculate sum of pixels in linereg
    Npix = width of linereg
    calculate rolling sum of Npix in contreg (from min+Npix/2 to max-Npix/2)
    calculate standard deviation of rolling sums
    if sum(linereg) > nsigma * stddev: line exists
    else: uplim = nsigma*stddev
    '''
    linespec = spec[(spec[wavekey]>linereg[0]) & (spec[wavekey]<linereg[1])]
    linesum = sum(linespec[fluxkey]-linespec[contkey])
    npix = len(linespec)
    
    if len(contreg) == 2:
        contspec = spec[(spec[wavekey]>contreg[0]) & (spec[wavekey]<contreg[1])]
    elif len(contreg) == 4:
        contspec = spec[(spec[wavekey]>contreg[0]) & (spec[wavekey]<contreg[1]) 
                        | (spec[wavekey]>contreg[2]) & (spec[wavekey]<contreg[3])]
    
    halfpix = np.ceil(npix/2.)
    lowhalf = int(np.floor(npix/2.))
    contmin = int(halfpix) # maybe add some extra thing here later for increased generality?
    contmax = int(len(contspec) - halfpix)
    
    sums = []
    for i in range(contmin, contmax):
        sum_i = sum(contspec[fluxkey][i-lowhalf:i+lowhalf] - contspec[contkey][i-lowhalf:i+lowhalf])
        sums.append(sum_i)
        
    median_sums = np.median(np.abs(sums))
    std_sums = np.std(np.abs(sums))
    
    if linesum >= nsigma*std_sums:
        print(f"There's a line here!! Significance: {linesum/std_sums:.2f}. Go forth and fit it!")
        return 0
    else:
        return nsigma*std_sums








# HERE BE DRAGONS
# actually just old versions of calculating upper limits
# the above function does a better job, so these are kept for posterity

#def is_line(spectrum, line_region, cont_region, n=3.):
#    '''
#    Check to see if a line is present
#    Takes an expected line region, and a continuum region
#    First does continuum substraction
#    Next takes standard deviation of continuum region
#    If the peak of the line region is > n*sigma above zero, we say there is a line, and we party!
#    '''
#    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1
#
#    spaxwave = spectrum.spectral_axis
#
#    fitted_cont = fit_continuum(spectrum, window=cont_region)
#    y_cont = fitted_cont(spaxwave)
#
#    contsub_spec = spectrum - y_cont
#    # maybe I should put all this continuum subtraction in a seperate function at some point? 
#    # could be better than redoing it every time...
#
#    # okay, take the std deviation of the continuum region!
#    noise_spec = extract_region(contsub_spec, cont_region, return_single_spectrum=True)
#    noiseflux = noise_spec.flux.value # should be a single array of flux values
#    noise_sigma = np.std(noiseflux)
#
#    # now we extract the line region
#    line_min, line_max = line_region.lower.value, line_region.upper.value
#
#    line_spec = extract_region(contsub_spec, line_region, return_single_spectrum=True)
#    line_peak = max(line_spec.flux.value)
#
#    # now, the decision! Will we take our talents to south beach?!?
#    if line_peak > n * noise_sigma: return True
#    else: return False


#def flux_upper_limit(spectrum, line_region, cont_region, n=3, width=0.00015, verbose=False):
#    '''
#    Check to see if a line is present, then if not set an upper limit on flux
#    Takes an expected line region, and a continuum region
#    First does continuum substraction
#    Next takes standard deviation of continuum region
#    If the peak of the line region is > n*sigma above zero, we say there is a line, and we party!
#    If the peak is less than n*sigma, we set an upper limit
#    We assume a Gaussian line profile with amplitude n*sigma, and width set by "width" parameter (0.0015 micron default)
#    that flux is our upper limit
#    Should probably test this - inject a 3-sigma peak and see where if it can be recovered/fit
#    that's a later Brian problem. Later Brian is so helpful. 
#    '''
#    cgs = u.erg * u.cm**-2 * u.s**-1 * u.AA**-1
#
#    spaxwave = spectrum.spectral_axis
#
#    fitted_cont = fit_continuum(spectrum, window=cont_region)
#    y_cont = fitted_cont(spaxwave)
#
#    contsub_spec = spectrum - y_cont
#    # maybe I should put all this continuum subtraction in a seperate function at some point? 
#    # could be better than redoing it every time...
#
#    # okay, take the std deviation of the continuum region!
#    noise_spec = extract_region(contsub_spec, cont_region, return_single_spectrum=True)
#    noiseflux = noise_spec.flux.value # should be a single array of flux values
#    noise_sigma = np.std(noiseflux)
#
#    # now we extract the line region
#    line_min, line_max = line_region.lower.value, line_region.upper.value
#
#    line_spec = extract_region(contsub_spec, line_region, return_single_spectrum=True)
#    line_peak = max(line_spec.flux.value)
#
#    # now, the decision! Will we take our talents to south beach?!?
#    if line_peak > n * noise_sigma: 
#        print("There's a line here! No upper limit needed! Yippee!")
#        return -1
#    else: 
#        max_flux = np.sqrt(2*np.pi) * (n*noise_sigma) * (width * 1e4) # amp = n*sigma, convert width to Angstrom from Micron
#        if verbose:
#            return max_flux, noise_sigma, line_peak
#        else:
#            return max_flux





