"""
Module containing functiosn to convert parameters to model images
Author: Andrew Miller <acm@seas.harvard.edu>
"""
import autograd.numpy as np
import CelestePy.util.mixture_profiles as mp
import util.dists.mog as mog_funs
from CelestePy.util.dists.mog import MixtureOfGaussians


## galaxy profile objects - each is a mixture of gaussians
galaxy_profs_dstn = [mp.get_exp_mixture(), mp.get_dev_mixture()]
galaxy_profs = [MixtureOfGaussians(means=g.mean, covs=g.var, pis = g.amp)
                for g in galaxy_profs_dstn]
galaxy_prof_dict = dict(zip(['exp', 'dev'], galaxy_profs))


def gen_point_source_psf_image(
        u,                         # source location in equatorial coordinates
        image,                     # FitsImage object
        xlim          = None,      # compute model image only on patch defined
        ylim          = None,      #   by xlim ylimcompute only for this patch
        check_overlap = True,      # speedup to check overlap before computing
        return_patch  = True,      # return the small patch as opposed to large patch (memory/speed purposes)
        psf_grid      = None,      # cached PSF grid to be filled out
        pixel_grid    = None       # Nx2 matrix of discrete pixel values to evaluate mog at
        ):
    """
    generates a PSF image (assigns density values to pixels) 
    Also known as gen_unit_flux_image
    """
    # compute pixel space location of source
    # returns the X,Y = Width, Height pixel coordinate corresponding to u

    # compute pixel space location, v_{n,s}
    v_s = image.equa2pixel(u)
    does_not_overlap = check_overlap and \
                       (v_s[0] < -50 or v_s[0] > 2*image.nelec.shape[0] or
                       v_s[1] < -50 or v_s[0] > 2*image.nelec.shape[1])
    if does_not_overlap:
        return None, None, None

    # create sub-image - make sure it doesn't go outside of field pixels
    if xlim is None and ylim is None:
        bound = image.R
        minx_b, maxx_b = max(0, int(v_s[0] - bound)), min(int(v_s[0] + bound + 1), image.nelec.shape[1])
        miny_b, maxy_b = max(0, int(v_s[1] - bound)), min(int(v_s[1] + bound + 1), image.nelec.shape[0])
        y_grid = np.arange(miny_b, maxy_b, dtype=np.float)
        x_grid = np.arange(minx_b, maxx_b, dtype=np.float)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
        pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))
    else:
        miny_b, maxy_b = ylim
        minx_b, maxx_b = xlim
        if pixel_grid is None:
            y_grid = np.arange(miny_b, maxy_b, dtype=np.float)
            x_grid = np.arange(minx_b, maxx_b, dtype=np.float)
            xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
            pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))
    grid_shape = (maxy_b-miny_b, maxx_b-minx_b)
    #psf_grid_small = gmm_like_2d(x       = pixel_grid,
    #                             ws      = image.weights,
    ##                             mus     = image.means + v_s,
    #                             sigs    = image.covars)
    psf_grid_small = np.exp(mog_funs.mog_loglike(pixel_grid,
                                            means = image.means+v_s,
                                            icovs = image.invcovars,
                                            dets  = np.exp(image.logdets),
                                            pis   = image.weights))

    # return the small patch and it's bounding box in the bigger fits_image
    if return_patch:
        return psf_grid_small.reshape(grid_shape, order='C'), \
               (miny_b, maxy_b), (minx_b, maxx_b)

    # instantiate a PSF grid 
    if psf_grid is None:
        psf_grid = np.zeros(image.nelec.shape, dtype=np.float)

    # create full field grid
    psf_grid[miny_b:maxy_b, minx_b:maxx_b] = \
        psf_grid_small.reshape(xx.shape, order='C')
    return psf_grid, (0, psf_grid.shape[0]), (0, psf_grid.shape[1])


def gen_galaxy_psf_image(th, u_s, img, xlim=None, ylim=None,
                                       check_overlap = True,
                                       unconstrained = True,
                                       return_patch=True):
    """
    generates the profile of a combination of exp/dev images.
    Calls the above function twice - once for each profile, and adds them
    together
    """
    # unpack shape params
    theta_s, sig_s, phi_s, rho_s = th[0:4]

    # generate unit flux model patch
    px, py = img.equa2pixel(u_s)
    galmix = MixtureOfGaussians.convex_combine(galaxy_profs,
                                               [theta_s, 1.-theta_s])
    Tinv  = gen_galaxy_transformation(sig_s, rho_s, phi_s, img.cd_at_pixel(px, py))
    amix  = galmix.apply_affine(Tinv, np.array([px, py]))
    cmix  = amix.convolve(img.psf)

    # compute bounding box
    if xlim is None and ylim is None:
        from util.bound.bounding_box import calc_bounding_radius
        bound = calc_bounding_radius(cmix.pis, cmix.means, cmix.covs,
                                     error=1e-5, center=np.array([px, py]))
        xlim = (np.max([0,                  np.floor(px - bound)]),
                np.min([img.nelec.shape[1], np.ceil(px + bound)]))
        ylim = (np.max([0,                  np.floor(py - bound)]),
                np.min([img.nelec.shape[0], np.ceil(py + bound)]))

    # compute values on grid
    return cmix.evaluate_grid(xlim, ylim), ylim, xlim


#########################################################################
# galaxy transformation helper functions - describe appearance of       #
# galaxy in two-d image from shape parameters                           #
#########################################################################

def gen_galaxy_ra_dec_basis(sig_s, rho_s, phi_s):
    '''
    Returns a transformation matrix that takes vectors in r_e
    to delta-RA, delta-Dec vectors.
    (adapted from tractor.galaxy)
    '''
    # convert re, ab, phi into a transformation matrix
    phi = (90. - phi_s) * np.pi / 180.  # np.deg2rad(90-phi_s)
    # convert re to degrees
    # HACK -- bring up to a minimum size to prevent singular
    # matrix inversions
    re_deg = max(1./30, sig_s) / 3600.
    cp = np.cos(phi)
    sp = np.sin(phi)
    # Squish, rotate, and scale into degrees.
    # resulting G takes unit vectors (in r_e) to degrees
    # (~intermediate world coords)
    return re_deg * np.array([[cp, sp*rho_s], [-sp, cp*rho_s]])


def gen_galaxy_transformation(sig_s, rho_s, phi_s, Ups_n):
    """ from dustin email, Jan 27
        sig_s (re)  : arcsec (greater than 0)
        rho_s (ab)  : axis ratio, dimensionless, in [0,1]
        phi_s (phi) : radians, "E of N", 0=direction of increasing Dec,
                      90=direction of increasing RAab = 
    """
    # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
    G = gen_galaxy_ra_dec_basis(sig_s, rho_s, phi_s)

    # "cd" takes pixels to degrees (intermediate world coords)
    cd = Ups_n

    # T takes pixels to unit vectors (effective radii).
    T    = np.dot(np.linalg.inv(G), cd)
    Tinv = np.linalg.inv(T)
    return Tinv

