import autograd.numpy as np

#############################################################################
# simple converter between mags, nanomaggies, and colors, and r-band ratio
# colors 
#############################################################################

def mags_to_nanomaggies(mags):
    return np.power(10., (mags - 22.5) / -2.5)


def nanomaggies_to_mags(nanos):
    return (-2.5) * np.log10(nanos) + 22.5


def make_color_flux_converters():

    # converter matrix -
    #create transformation matrix that takes log [u g r i z] fluxes, 
    # and turns them into [lr, lu - lg, lg - lr, lr - li, li - lz]
    #
    # the mixture of gaussians is then a law on this transformed 
    # space
    #
    #  [lu lg lr li lz] dot [ 0   1   0   0   0
    #                         0  -1   1   0   0
    #                         1   0  -1   1   0
    #                         0   0   0  -1   1
    #                         0   0   0   0  -1 ]
    A = np.zeros((5,5))
    A[2,0] = 1.
    A[0,1] = A[1,2] = A[2,3] = A[3,4] = 1.
    A[1,1] = A[2,2] = A[3,3] = A[4,4] = -1.
    Ainv = np.linalg.inv(A)

    def fluxes_to_colors(fluxes):
        return np.dot(np.log(fluxes), A)

    def colors_to_fluxes(colors):
        return np.exp(np.dot(colors, Ainv))

    return fluxes_to_colors, colors_to_fluxes

fluxes_to_colors, colors_to_fluxes = make_color_flux_converters()


def mags_to_colors(mags):
    """ ugriz magnitudes to "color" - which are neighboring ratios
        colors = [r, u - g, g - r, r - i, i - z]
    """
    return fluxes_to_colors(mags_to_nanomaggies(mags))

def colors_to_mags(colors):
    """ takes [u-g, g-r, r-i, i-z, r] vector of colors to ugriz magnitudes"""
    return nanomaggies_to_mags(colors_to_fluxes(colors))


#create transformation matrix that takes log [u g r i z] fluxes, 
# and turns them into [lu - lr, lg - lr, li - lr, lz - lr, lr]
#
# the mixture of gaussians is then a law on this transformed 
# space
#
#  [u g r i z] dot [ 1   0   0   0   0
#                    0   1   0   0   0
#                   -1  -1  -1  -1   1
#                    0   0   1   0   0
#                    0   0   0   1   0
#

Ar = np.zeros((5,5))
Ar[0,0] = Ar[1,1] = Ar[3,2] = Ar[4,3] = 1.
Ar[2, :]  = -1
Ar[2, -1] = 1.
Arinv = np.linalg.inv(Ar)

def mags_to_rcolors(mags, ridx = 2):
    """ takes a vector of ugriz mags, and turns into colors:
        [lu - lr, lg - lr, li - lr, lz - lr, lr]
    """
    return np.dot(mags, A)

def rcolors_to_mags(colors):
    return np.dot(colors, Ainv)


###################################################
# Soft galaxy shape parameterization              #
###################################################

def rAbPhiToESoft(r, ba, phi):
    ab    = 1./ba
    e     = (ab - 1) / (ab + 1)
    ee    = -np.log(1 - e)
    angle = np.deg2rad(phi)
    ee1   = ee * np.cos(angle)
    ee2   = ee * np.sin(angle)
    return (np.log(r), ee1, ee2)

def eSoftToRAbPhi(logr, ee1, ee2):
    r  = np.exp(logr)
    ee    = np.sqrt(ee1*ee1 + ee2*ee2)
    angle = np.arccos(ee1 / ee)
    phi   = np.rad2deg(angle)
    e  = 1 - np.exp(-ee)
    ab = - (1. + e) / (e - 1)
    return r, 1./ab, phi

def unconstrain_gal_shape(shape):
    """ unconstrains galaxy shape parameters
    Args: 
        shape: tuple/list with parameters: 
                - theta : dev/exp mix
                - sigma : radial extent in arcsec
                - phi   : angle of major axis, east of north in [0, 180] deg
                - rho   : ratio of minor to major axis, in [0, 1]
    Returns: 
        lshape : tuple/list that can be inverted w/ constrain_gal_shape
    """
    theta, sigma, phi, rho = shape
    logit_theta  = np.log(theta) - np.log(1. - theta)
    lr, ee1, ee2 = rAbPhiToESoft(sigma, rho, phi)
    return np.array([logit_theta, lr, ee1, ee2])

def constrain_gal_shape(lshape):
    """ Constrains an unconstrained galaxy shape """
    lg_theta, lr, ee1, ee2 = lshape
    theta       = 1./(1. + np.exp(-lg_theta))
    r, rho, phi = eSoftToRAbPhi(lr, ee1, ee2)
    return np.array([theta, r, phi, rho])


if __name__=="__main__":

    # check inverses
    r = 36.
    ba = .6
    phi = 95
    lr, ee1, ee2 = rAbPhiToESoft(r, ba, phi)
    r1, ba1, phi1 = eSoftToRAbPhi(lr, ee1, ee2)

    # check flux => color; color => flux
    fluxes = np.array([1.25, 9.6, 24.4, 33.07, 40.85])
    cvec = fluxes_to_colors(fluxes)
    fvec = colors_to_fluxes(cvec)

    mags = nanomaggies2mags(fluxes)
    print mags[2], cvec[-1]




