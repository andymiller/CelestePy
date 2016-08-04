"""
Various MCMC techniques to infer star parameters.  Model and functions are
described in CelestePy.util.point_source
"""
import autograd.numpy as np
from autograd import grad, hessian
from CelestePy.point_source import make_gen_model_image, \
                                   make_lnpdf_fun, bands
import synthetic_constants as sc
from CelestePy.util.transform import fluxes_to_colors


def run_star_inference(img_dict, img_constants,
                       img_masks             = None,
                       num_chains            = 8,
                       num_samples_per_chain = 200,
                       n_jobs                = None,
                       synthetic_images = False,
                       true_params      = None,
                       eps_dict         = None ):
    """
    Args:

        img_constants : dictionary indexed by UGRIZ, each entry is a dict with

            ic['phi']: reference RA,DEC
            ic['rho']: reference pixel (aligned with phi)
            ic['Ups']: transformation matrix that takes pixel coordinates
                       to equa coordinates
                        ra, dec = phi + Ups ([px,py] - rho)
                       see the function CelestePy.fits_image.pixel_to_equa

            ic['band']: band for these image constants (should be same as index)
            ic['photons_per_nmgy']: average number of photons expected for
                                    each nmgy of flux
            ic['psf'] : Mixture of gaussians representing the point spread function

    """
    # create image-specific rendering functions
    img_funs  = { k: make_gen_model_image(
                            phi              = ic['phi'],
                            rho              = ic['rho'],
                            Ups_inv          = ic['Ups_inv'],
                            psf              = ic['psf'],
                            band             = ic['band'],
                            photons_per_nmgy = ic['photons_per_nmgy'])
                       for k, ic in img_constants.iteritems() }
    model_img_funs       = { k: img_funs[k][0] for k in bands }
    brightest_funs       = { k: img_funs[k][1] for k in bands }
    sample_funs          = { k: img_funs[k][2] for k in bands }
    model_img_fixed_funs = { k: img_funs[k][3] for k in bands }

    # create img dict or pass it in
    if synthetic_images:
        assert true_params is not None and eps_dict is not None
        imgdict = {}
        for b in bands:
            nx, ny     = img_constants[b]['shape']
            xx, yy     = np.meshgrid(np.arange(nx), np.arange(ny))
            pixel_grid = np.column_stack([xx.flatten(), yy.flatten()])
            imgdict[b] = sample_funs[b](true_params, eps_dict[b],
                                        pixel_grid).reshape(xx.shape)

    # first find the brightest pixel location
    us = np.array([brightest_funs[b](imgdict[b]) for b in bands])
    ubar, ustd = us.mean(0), us.std(0)

    # construct lnpdf and prior sample fun
    lnpdf, sample_from_prior, lnpdf_u_maker = \
        make_lnpdf_fun(imgdict, eps_dict,
                       model_img_funs = model_img_funs,
                       u_guess = ubar,
                       u_error = ustd,
                       pixel_grid = pixel_grid, 
                       psf_image_fixed_location_makers = model_img_fixed_funs)

    # construct sampler steps - a mix of gibbs sampling steps
    from CelestePy.util.infer.mcmc import mcmc_multi_chain
    from CelestePy.util.infer.mh import mhstep
    from CelestePy.util.infer.slicesample import slicesample
    def gibbs_step(params, llth):
        colors, u = params[0:5], params[5:7]
        lnpdfu = lnpdf_u_maker(u)
        for _ in range(5):
            colors, _ = slicesample(colors, lnpdfu, compise=True)
        u, ll     = slicesample(u, lambda u: lnpdf(np.concatenate([colors, u])), compwise=False)
        return np.concatenate([colors, u]), ll

    # set up initial values for each chain
    gibbs_funs = [gibbs_step for _ in xrange(num_chains)]
    th0s       = 1.1 * np.array([sample_from_prior() for _ in xrange(num_chains)])
    ll0s       = np.array([ lnpdf(th) for th in th0s ])
    th_samps, ll_samps = \
        mcmc_multi_chain(th0s, ll0s, gibbs_funs,
                         Nsamps=num_samples_per_chain, burnin=100, n_jobs=n_jobs)
    _, Nsamps, D = th_samps.shape

    # return samples, organized by each chain
    return th_samps, ll_samps


def run_synthetic_star_inference(noise_scale=1.):

    # true params to be inferred
    true_params, true_colors = sc.synthetic_star_params()

    # image noise params
    eps_dict = {'u': 28., 'g': 307., 'r': 684., 'i': 817, 'z': 484.}
    eps_dict = {b: noise_scale*e for b,e in eps_dict.iteritems()}

    ######################################################################
    # set up image information - image constants need to be specified    #
    # from fits files                                                    #
    ######################################################################
    img_constants = sc.img_constants
    for k, ic in sc.img_constants.iteritems():
        ic['rho']   = np.array([25., 25.])
        ic['phi']   = img_constants['r']['phi'] # make all images have the same reference RA/DEC
        ic['shape'] = (50, 50)

    # run inference
    th_samps, ll_samps = \
        run_star_inference(img_dict = None,
                           img_constants = img_constants,
                           num_chains = 8,
                           num_samples_per_chain = 500,
                           synthetic_images = True,
                           true_params      = true_params,
                           eps_dict         = eps_dict, 
                           n_jobs           = None)

    return th_samps, ll_samps


if __name__=="__main__":

    # for a handful of noise scales, run star parameter inference
    noise_scales = [.1, .5, 1., 2., 4., 8., 16.]

    from joblib import Parallel, delayed
    res_list = Parallel(n_jobs=len(noise_scales), verbose=1)(
        delayed(run_synthetic_star_inference)(ns) for ns in noise_scales)

    # zip up, and save file
    noise_scale_res = dict(zip(noise_scales, res_list))
    import cPickle as pickle
    with open('star_noise_scale_results.pkl', 'wb') as f:
        pickle.dump(noise_scale_res, f)

