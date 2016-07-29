"""
Star generative model

    Equatorial location:
        u_s  = [ra, dec] ~ Unif[0, 1]
    Reference band flux:
        r_s  | star ~ Gamma(a_red_star, b_red_star)
    Colors:
        c_sb | star = log(brightness_b) - log(brightness_b+1)
        e.g. for SDSS ugriz: c_s = c_ug, c_gr, c_ri, c_iz

    Image-specific distortions (indexed by n):
        eps_n   : img noise
        kappa_n : num photons a pixel is expected to record per one nanomaggie
                  e.g. pixel m has nmgy illumination b_m, then pixel x_m is
                        x_m ~ Pois(lam_m = b_m * kappa_n)
        psf_n   : point spread function in pixel-space.  given a photon that 
                  is emitted at pixel m in the ideal sky view, the psf
                  describes the probability that the pixel is observed at pixel
                  m' as
                        psf_n(m'; m, \theta_n)
                  our psf's are mixtures of 3 gaussians, but these functions
                  could be approximated in other ways. \theta = (pi, mus, Sigs)
                  in the MoG case

    Observations: for a single source s, and iamge n, we model each pixel as
                  conditionally poisson
            flux_{s,n} = flux of s in band_n (from reference or from colors)
            lam_{n,m}  = eps_n + psf_n(m; u, theta_n) * flux_n * kappa_n
            x_{n,m}    ~ Pois(lam_{n,m})

    Background source: some sources may overlap with source s.  We first
                       proceed by assuming we know exactly their contribution
                       to the background.  The new lambda would be

            lam_{n,m} = eps_n + background_{n,m} + psf_n(m; u) * flux_n * kappa_n

                        where background_n = sum_{s'=!s} psf_n(m;u_s') * flux_{

    The inferential goal is to characterize the posterior distribution

        u_s, r_s, c_s | X
"""
from autograd import grad, hessian
import autograd.numpy as np
import CelestePy.fits_image as fi
from CelestePy.util.dists.poisson import poisson_logpmf
from CelestePy.util.transform import colors_to_fluxes

bands = ['u', 'g', 'r', 'i', 'z']


def make_gen_model_image(phi, rho, Ups_inv, psf, band, photons_per_nmgy):
    """ Monad for creating model images, caching image specific constants
    Args:
        phi               : [ra, dec] reference point
        rho               : [px, py] corresponding pixel px, py reference point
        Ups_inv           : inverse transformation
        psf               : image-specific point spread function
        band              : u,g,r,i,z band for the image
        photons_per_nmgy  : conversion between expected photons and nmgy's for this image
    """

    # make image equatorial to pixel conversion functions specific to
    # img params passed in
    equa2pixel = lambda u: fi.equa2pixel(u, phi_n=phi,
                                            Ups_n_inv=Ups_inv, rho_n=rho)
    pixel2equa = lambda p: fi.pixel2equa(p, phi_n = phi,
                                            Ups_n = np.linalg.inv(Ups_inv),
                                            rho_n=rho)

    def gen_psf_image(params, pixel_grid):
        """ Generates a psf image given params = (colors, u) on a 
        fixed set of pixels (using the image parameters set in
        make_gen_model_image """
        # unpack params
        colors, u = params[0:5], params[5:7]
        # unit flux img
        unit_flux_img = gen_unit_flux_image(u, pixel_grid)
        return scale_unit_flux_image(unit_flux_img, colors)

    def gen_unit_flux_image(u, pixel_grid):
        """ generate a unit flux image corresponding to the image parameters
        defined in this closure """
        pix_u = equa2pixel(u)
        return psf.pdf(pixel_grid, mean_shift=pix_u)

    def scale_unit_flux_image(unit_flux_img, colors):
        """ given a unit_flux_img, and colors, convert colors to fluxes, scale
        them by the image specific constant, and return product"""
        # grab flux appropriate to image
        fluxes = colors_to_fluxes(colors)
        flux_n = fluxes[ bands.index(band) ]
        mimg   = flux_n * photons_per_nmgy * unit_flux_img
        return mimg

    def gen_psf_image_fixed_location_maker(u, pixel_grid):
        """ given a u and a pixel_grid, returns a function that generates
        a model image given only colors - for quicker inference over fluxes """
        unit_flux_image = gen_unit_flux_image(u, pixel_grid)
        def gen_psf_image_fixed_u(colors):
            return scale_unit_flux_image(unit_flux_image, colors)
        return gen_psf_image_fixed_u


    def get_brightest_radec(img):
        """ given an image with the closure's specification, return the
        brightest pixel in ra, dec """
        ys, xs = np.where(img == img.max())
        pxy = np.array([xs[0], ys[0]])
        return pixel2equa(pxy)

    # TODO add background image
    def sample_image(params, eps, pixel_grid, background_image=0.):
        mimg = gen_psf_image(params, pixel_grid) + eps + background_image
        return np.random.poisson(mimg)

    return gen_psf_image, get_brightest_radec, \
           sample_image, gen_psf_image_fixed_location_maker


def make_lnpdf_fun( imgdict, eps_dict,
                    model_img_funs,
                    u_guess, u_error, pixel_grid,
                    psf_image_fixed_location_makers,
                    background_img_dict = None):
    """ returns log posterior functions that can be used for finding map or
    within a MCMC routine 

    Args:
        - imgdict  : dict of photon count images, indexed by band, imgdict['r'] = NxM array
        - eps_dict : noise parameters for each
        - model_img_funs: dict of functions that create a model image,
                            model_img_funs['r'](params, pixel_grid) = r_mimg
        - u_guess, u_error: guess where the center of the object should be,
                            (based on brightest pixel, old catalog)
        - pixel_grid : list of (x, y) values represented by imgdict images
        - psf_image_fixed_location_makers: dict of functions, each of which
                take in a u=(ra,dec) and a pixel_grid, and return a function
                that generates a model image given a color (holding fixed the 
                u already passed in).  This function passing intends to cache
                a unit_psf_image, allowing for the quick lnpdf calculation
                of the color parameters (so they can be iterated over in a
                slice-within-gibbs sort of MCMC step)
        - background_img_dict: dict of images that add to the model image
            before computing the likelihood.  
            TODO: combine this with 'eps_dict'
    """
    def lnpdf(params):
        ll = 0.
        for b, img in imgdict.iteritems():
            mimg = model_img_funs[b](params, pixel_grid) + eps_dict[b]
            ll += np.mean(poisson_logpmf(img.ravel(), mimg))
        if np.isnan(ll):
            return -np.inf
        return ll + logprior(params)

    # load prior over colors
    star_color_mog = load_color_prior()

    def logprior(params):
        colors, u = params[0:5], params[5:7] - u_guess
        return    star_color_mog.logpdf(colors) \
               - .5*np.dot(u/u_error, u/u_error)

    def sample_from_prior():
        color_samp = star_color_mog.rvs(size=1)[0]
        u_samp = u_guess + np.random.randn(2) * u_error
        return np.concatenate([color_samp, u_samp])

    # pdf conditional on a location - for quicker updates of colors
    def lnpdf_fixed_u_maker(u):
        # for each image create a function that generates a psf image with a
        # fixed u (caches the unit_flux_image in a closure)
        gen_psf_fixed_u_funs = \
            { b: make_fun(u, pixel_grid)
              for b, make_fun in psf_image_fixed_location_makers.iteritems() }
        def lnpdf_fixed_u(colors):
            """ returns lnpdf of a collection of images conditioned on a fixed
            location u """
            ll = 0.
            for b, img in imgdict.iteritems():
                mimg = gen_psf_fixed_u_funs[b](colors) + eps_dict[b]
                ll += np.mean(poisson_logpmf(img.ravel(), mimg))
            if np.isnan(ll):
                return -np.inf
            return ll + star_color_mog.logpdf(colors)
        return lnpdf_fixed_u

    # initialize a value for theta
    return lnpdf, sample_from_prior, lnpdf_fixed_u_maker


def load_color_prior():
    """ loads a mixture of gaussians object from CelestePy/model_data/...
    """
    import CelestePy.util.dists.mog as mog
    import CelestePy, os
    import cPickle as pickle
    prior_param_dir = os.path.join(os.path.dirname(CelestePy.__file__),
                                                   'model_data')
    star_color_mog  = pickle.load(open(
        os.path.join(prior_param_dir, 'star_colors_mog.pkl'), 'rb'))
    return star_color_mog

