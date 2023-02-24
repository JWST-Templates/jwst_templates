''' Helper functions to analyze data from the JWST TEMPLATES Early Release Science
(ERS) program. See our website: https://sites.google.com/view/jwst-templates/?pli=1 
and our code https://github.com/JWST-Templates'''


def get_targnames():
    # Let's use common nomenclature everywhere to make life easier
    return(('SGAS1723+34', 'SGAS1226+21', 'SPT2147-50', 'SPT0418-47'))

def get_redshifts(kind='nebular'):
    names = get_targnames()
    if kind == 'nebular' :
        redshifts = [1.3293, 2.9260, 3.76, 4.2248] # these last 2 need more precision
    elif kind == 'stellar'
        redshifts = [1.3293, 2.9252, 3.76, 4.22]
    # Origin of these redshifts:
    #    SGAS1723: Rigby et al. 2021
    #    SGAS1226: Rigby et al. 2018 
    # 
    return( dict(zip(names, redshifts)))
