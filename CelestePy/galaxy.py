"""
Galaxy generative model

    Equatorial location:
        u_s  = [ra, dec] ~ Unif[0, 1]
    Reference band flux:
        r_s  | star ~ Gamma(a_red_star, b_red_star)
    Colors:
        c_sb | star = log(brightness_b) - log(brightness_b+1)
        e.g. for SDSS ugriz: c_s = c_ug, c_gr, c_ri, c_iz
    Shape:
        theta_s ~ p(theta_s) in [0, 1].  mix between devac (0) and exp (1) galaxy profile
        phi_s   ~ rotation angle, EAST OF NORTH [0, 180]
        rho_s   ~ minor/major axis ratio [0, 1]
        sig2_s  ~ effective radius (arc secs) > 0

    "Ideal sky view" appearance:
        f_exp(m), f_dev(m) = exp and dev profile functions (pixels)
        R_s = [[cos phi_s - sin phi_s],
               [sin phi_s   cos phi_s]]
        W_s = R_s^T diag([sig2_s, sig2_s rho_s]) R_s
        f_s(m) = theta_s * f_exp(m, W_s) + (1 - theta_s) * f_dev(m, W_s)
        f_exp(m, W_s) = \sum_j nu_j N(m; u_s, nu_j W_s)

        f_s(m) needs to be convolved w/ the image specific PSF to explain
        observations

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
            lam_{n,m}  = eps_n + f_s(m; u, W_s) * flux_n * kappa_n
            x_{n,m}    ~ Pois(lam_{n,m})

    Background source: some sources may overlap with source s.  We first
                       proceed by assuming we know exactly their contribution
                       to the background.  The new lambda would be

            lam_{n,m} = eps_n + background_{n,m} + f_s(m; u, W_s) * flux_n * kappa_n

                        where background_n = sum_{s'=!s} psf_n(m;u_s') * flux_{

    The inferential goal is to characterize the posterior distribution

        u_s, r_s, c_s | X
"""
from autograd import grad, hessian
import autograd.numpy as np
import CelestePy.fits_image as fi
from CelestePy.util.dists.poisson import poisson_logpmf
from CelestePy.util.transform import colors_to_fluxes, \
                                     unconstrain_gal_shape, constrain_gal_shape
import CelestePy.celeste_funs as celeste
import CelestePy.util.mixture_profiles as mp
import CelestePy.util.dists.mog as mog

bands             = ['u', 'g', 'r', 'i', 'z']
galaxy_profs_dstn = [mp.get_exp_mixture(), mp.get_dev_mixture()]
galaxy_profs      = [mog.MixtureOfGaussians(means=g.mean, covs=g.var, pis = g.amp)
                     for g in galaxy_profs_dstn]
galaxy_prof_dict  = dict(zip(['exp', 'dev'], galaxy_profs))


def make_gen_model_image(phi, rho, Ups_inv, psf, band, photons_per_nmgy):
    """ Monad for creating model images, caching image specific constants
    Args:
        phi               : [ra, dec] reference point
        rho               : [px, py] corresponding pixel px, py reference point
        Ups_inv           : inverse transformation
         or Ups           : 2x2 matrix: projection matrix from pixel to
                            equa coordinates (to go along w/ phi and rho)
        psf               : image-specific point spread function
        band              : u,g,r,i,z band for the image
        photons_per_nmgy  : conversion between expected photons and nmgy's for this image
    """

    # make image equatorial to pixel conversion functions specific to
    # img params passed in
    Ups        = np.linalg.inv(Ups_inv)
    equa2pixel = lambda u: fi.equa2pixel(u, phi_n=phi,
                                            Ups_n_inv=Ups_inv, rho_n=rho)
    pixel2equa = lambda p: fi.pixel2equa(p, phi_n = phi,
                                            Ups_n = Ups,
                                            rho_n=rho)

    def gen_psf_image(params, pixel_grid):
        """ Generates a psf image given params = (colors, u) on a 
        fixed set of pixels (using the image parameters set in
        make_gen_model_image """
        # unpack params
        colors, u, shape = params[0:5], params[5:7], params[7:]
        # unit flux img
        unit_flux_img = gen_unit_flux_image(u, shape, pixel_grid)
        return scale_unit_flux_image(unit_flux_img, colors)

    def gen_rotation_mat(phi):
        return np.array([ [ np.cos(phi), -np.sin(phi) ],
                          [ np.sin(phi),  np.cos(phi) ] ])

    def construct_components(theta_s, u, W):
        means = []
        covs  = []
        pis   = []
        for k in xrange(len(psf.pis)):
            for i, th in enumerate([theta_s, 1. - theta_s]):
                prof = galaxy_profs[i]
                for j in xrange(len(prof.pis)):
                    pi  = psf.pis[k] * th * prof.pis[j]
                    mu  = u + psf.means[k]
                    cov = psf.covs[k] + np.dot(prof.covs[j], W)
                    means.append(mu)
                    pis.append(pi)
                    covs.append(cov)
        means = np.array(means)
        covs = np.array(covs)
        pis  = np.array(pis)
        return mog.MixtureOfGaussians(means=means, covs=covs, pis=pis)

    def construct_components_vec(theta_s, u, W):
        pass

    def print_shape(shape):
        cshape = constrain_gal_shape(shape)
        th, sig, phi, rho = cshape
        print "theta", th
        print "sig  ", sig
        print "phi  ", phi
        print "rho  ", rho

    def gen_unit_flux_image(u, shape, pixel_grid):
        """ generate a unit flux image corresponding to the image parameters
        defined in this closure """
        # generate unit flux model patch
        pix_u = equa2pixel(u)
        theta_s, sig_s, phi_s, rho_s = constrain_gal_shape(shape)

        # ideal sky view of galaxy
        # dev/exp mixture of gaussians
        #galmix = mog.MixtureOfGaussians.convex_combine(galaxy_profs,
        #                                               [theta_s, 1.-theta_s])
        # TODO this appearance needs to incorporate the Upsilon transformation
        R = gen_rotation_mat(np.deg2rad(180. - phi_s))
        D = np.diag(np.array([sig_s, sig_s*rho_s]))
        W = np.dot(np.dot(R.T, D), R)
        try:
            cmix = construct_components(theta_s, pix_u, W)
        except Exception as e:
            print_shape(shape)
            print e
            raise Exception

        # convolve with PSF to get 
        #amix = galmix.apply_affine(Wn, pix_u)
        #cmix  = amix.convolve(psf)
        return cmix.pdf(pixel_grid)

    def scale_unit_flux_image(unit_flux_img, colors):
        """ given a unit_flux_img, and colors, convert colors to fluxes, scale
        them by the image specific constant, and return product"""
        # grab flux appropriate to image
        fluxes = colors_to_fluxes(colors)
        flux_n = fluxes[ bands.index(band) ]
        mimg   = flux_n * photons_per_nmgy * unit_flux_img
        return mimg

    def gen_psf_image_fixed_unit_image_maker(u, shape, pixel_grid):
        """ given a u and a pixel_grid, returns a function that generates
        a model image given only colors - for quicker inference over fluxes """
        unit_flux_image = gen_unit_flux_image(u, shape, pixel_grid)
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
           sample_image, gen_psf_image_fixed_unit_image_maker, gen_unit_flux_image


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
        lprior = logprior(params)
        if lprior == -np.inf:
            return lprior
        ll = 0.
        for b, img in imgdict.iteritems():
            mimg = model_img_funs[b](params, pixel_grid) + eps_dict[b]
            ll += np.mean(poisson_logpmf(img.ravel(), mimg))
        if np.isnan(ll):
            return -np.inf
        return ll + lprior

    # load prior over colors
    color_mog = load_color_prior()

    def lngamma(x, shape, scale):
        return ((shape - 1.)*np.log(x) - sig2/scale)

    def logprior(params):
        colors, u, shape = params[0:5], params[5:7] - u_guess, params[7:]
        th, sig2, phi, rho = constrain_gal_shape(shape)
        gam_shape, gam_scale = 2., 3.
        bet_a, bet_b = 2., 2.
        if sig2 > 10000 or rho < .0001 or rho > .9999 or phi < 0.00001 or phi > 179.9999:
            return -np.inf
        return    color_mog.logpdf(colors) \
               - .5*np.dot(u/u_error, u/u_error) \
               - (.5 / 25.) * np.log(sig2)**2 \
               + ((bet_a-1) * np.log(rho) + (bet_b-1) * np.log(1 - rho)) \
               - .5 * shape[0]**2

    def sample_from_prior():
        color_samp = color_mog.rvs(size=1)[0]
        u_samp = u_guess + np.random.randn(2) * u_error
        shape_samp = np.array([ np.random.rand(),        # theta
                                np.exp(1.8*np.random.randn()),   # sig2
                                np.random.rand() * 180., # phi
                                np.random.rand()         # rho
                              ])
        return np.concatenate([color_samp, u_samp,
                               unconstrain_gal_shape(shape_samp)])

    # pdf conditional on a location - for quicker updates of colors
    def lnpdf_fixed_u_maker(u, shape):
        # for each image create a function that generates a psf image with a
        # fixed u (caches the unit_flux_image in a closure)
        gen_psf_fixed_u_funs = \
            { b: make_fun(u, shape, pixel_grid)
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
            return ll + color_mog.logpdf(colors)
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
    gal_color_mog  = pickle.load(open(
        os.path.join(prior_param_dir, 'gal_colors_mog.pkl'), 'rb'))
    return gal_color_mog

