'''

Masking code to be used with various instruments' datasets.

For now, this holds the NIRSpec masking code, but we've made it separate than the spec.py module so that it can be expanded to include masking codes for the other instruments.




'''
from jwst_templates import spec



def makemask_NIRSpec(dic,dims,wave_range,line_params,snr=5,verbose=False):
    '''
    dic -- the input dictionary of spectra, assumes that
           each spectrum has columns wave, flux, ferr, continuum
    dims -- the dimensions of the FITS cube
    wave_range -- the wavelength range around the emission line
    line_params -- the input parameters of the emission line
    snr -- the lower threshold to use in making map; default 5
    verbose -- if True, will print the number of failed fits
    
    '''
    # getting key coordinates of dictionary
    coords = list(dic.keys())
    
    # checking that dictionary dataframes have a "continuum" column
    # if not this function fails and prints out the error message
    assert 'continuum' in spaxels[coords[0]], 'Dictionary of dataframes need to have a continuum column for this function to work.  Use the spec.fit_autocont() function on the dictionary spectra and then return to this make_mask() function.'
    
    
    # making a 2D SNR array
    linemap = np.zeros((1,dims[1],dims[2]))
    linemap[:] = np.nan # starting out with an array of NaNs
    
    error_count = 0 # keeping track of the except's thrown
    
    if verbose == True: print('Running through spaxel coordinates.',end='\n\n')
    
    # running through coordinates
    # -- splitting them by the , for the lineflux map part
    for x,y in coords.split(','): 

        # pulling spectra from the dictionary of spectra
        # where the dictionary key is the spaxel coordinates
        spectrum = dic[f'{x},{y}'].copy()
        spectrum = spec.convert_jwst_to_cgs(spectrum)
        
        # subtracting continuum
        spectrum['cont_sub'] = spectrum.flux.values - spectrum.continuum.values
        
        # fitting chosen bright emission line
        flux, ferr = spec.cont_sub_curvefit(spectrum,wave_range,line_params)

        # saving SNR value to map
        linemap[0,int(y),int(x)] = flux/ferr
    
    if verbose == True: print(f'Setting all spaxels with S/N < {snr} to NaN.',end='\n\n')
    
    linemap[linemap<snr] = np.nan
        
    return linemap

