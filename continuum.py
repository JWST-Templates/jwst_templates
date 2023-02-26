# Helper functions for fitting continua to JWST spectra, from the TEMPLATES ERS team
# Contributors: jrigby, 2023
import pandas
import astropy.convolution
import numpy as np



def load_default_linelist(v2mask=200.):
    # Load a linelist of rest-frame optical (need to add near-IR, incl Paa,B) emission
    # lines, to be masked before fitting the continuum.
    # column v2mask' is velocity +- to mask each emission line, in km/s.  May want to adjust for the brightest lines.
    # Does not work yet, still in dev. -jrigby***
    linelistfile = '/Users/jrrigby1/Python/TEMPLATES/jwst_templates/emission_lines.txt'
    # above is hardcoded.  Need to train it to look in the python module. **
    LL = pandas.read_csv(linelistfile, comment='#', delim_whitespace=True)
    LL['v2mask'] = v2mask
    return(LL)


def round_up_to_odd(f):
    # Boxcar needs to be an odd number
    return np.ceil(f) // 2 * 2 + 1

def get_boxcar4autocont(sp, smooth_length=100., rest_disp=2.0) :
    # Helper function for fit_autocont().  For the given input spectrum, finds the number of pixels that
    # corresponds to smooth_length in rest-frame Angstroms.  This will be the boxcar smoothing length.
    # smooth_length is smothing length in rest-frame Angstroms.  Default of 100A works well for MagE spectra.
    # rest_disp is rest-frame dispersion in Angstroms
    return(np.int(util.round_up_to_odd(smooth_length / rest_disp)))  # in pixels


def fit_autocont(sp, zz, LL=None, colv2mask='v2mask', v2mask=200., boxcar=1001, flag_lines=True, colwave='wave', colf='fnu', colmask='contmask', colcont='fnu_autocont') : 
    ''' Automatically fits a smooth continuum to a spectrum.  Generalized from Jane's version for Magellan/MagE data.
     Inputs:  sp,  a Pandas data frame containing the spectra, 
              zz,  the systemic redshift.
              LL,  a Pandas data frame containing the linelist to mask, opened by load_default_linelist() or similar.
              colv2mask,  column containing velocity +- to mask each line, km/s. 
              v2mask,     that velocity in km/s
              boxcar,   size of the boxcar smoothing window, in pixels
              flag_lines (Optional) flag, should the fit mask out known spectral features? Can turn off for debugging
              colwave, colf:  which columns to find wave, flux/flam/fnu
              colmask,  the column that has the mask, of what should be ignored in continuum fitting.  True is masked
              colcont, column to write the continuum.  The output!
    '''
    if LL == None:  LL = load_default_linelist(v2mask=v2mask)  # Grab a default linelist if none provided

    if flag_lines :
        if colmask not in sp : sp[colmask] = False
        flag_near_lines(sp, LL, colv2mask=colv2mask, colwave=colwave)  # lines are masked in sp.linemask
        sp.loc[sp['linemask'], colmask] = True           # add masked lines to the continuum-fitting mask
    # astropy.convolve barfs on pandas.Series as input.  Use .to_numpy() to send as np array
    smooth1 = astropy.convolution.convolve(sp[colf].to_numpy(), np.ones((boxcar,))/boxcar, boundary='extend', fill_value=np.nan, mask=sp[colmask].to_numpy()) # boxcar smooth
    small_kern = int(round_up_to_odd(boxcar/10.))
    smooth2 = astropy.convolution.convolve(smooth1, np.ones((small_kern,))/small_kern, boundary='extend', fill_value=np.nan) # Smooth again, to remove nans
    sp[colcont] = pandas.Series(smooth2)  # Write the smooth continuum back to data frame
    sp[colcont].interpolate(method='linear',axis=0, limit_direction='both', inplace=True)  #replace nans
    #print "DEBUGGING", np.isnan(smooth1).sum(),  np.isnan(smooth2).sum(), sp[colcont].isnull().sum()
    return(smooth1, smooth2)  # Do we need to return both? This was debugging


