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
import synthetic_constants as sc
import CelestePy.fits_image as fi
from CelestePy.util.dists.poisson import poisson_logpmf
from CelestePy.util.transform import fluxes_to_colors, colors_to_fluxes

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
        # unpack params
        colors, u = params[0:5], params[5:7]

        # unit flux image
        pix_u = equa2pixel(u)
        unit_flux_img = psf.pdf(pixel_grid, mean_shift=pix_u)

        # grab flux appropriate to image
        fluxes = colors_to_fluxes(colors)
        flux_n = fluxes[ bands.index(band) ]
        mimg   = flux_n * photons_per_nmgy * unit_flux_img
        return mimg

    return gen_psf_image, equa2pixel, pixel2equa


def sample_image(params, gen_psf_image, eps, pixel_grid):
    mimg = gen_psf_image(params, pixel_grid) + eps
    return np.random.poisson(mimg)


def load_color_prior():
    import CelestePy.util.dists.mog as mog
    import CelestePy, os
    import cPickle as pickle
    prior_param_dir = os.path.join(os.path.dirname(CelestePy.__file__),
                                                   'model_data')
    star_color_mog  = pickle.load(open(
        os.path.join(prior_param_dir, 'star_colors_mog.pkl'), 'rb'))
    return star_color_mog


def get_brightest_location(imgdict, pixel2equa_funs):
    def get_radec_max(band):
        img = imgdict[band]
        pxy = np.where(img == img.max())[0]
        return pixel2equa_funs[band](pxy)
    us = np.array([get_radec_max(b) for b in bands])
    #TODO weight this by noise level of each image ...
    return us.mean(0), us.std(0)


def make_lnpdf_fun(imgdict, eps_dict, model_img_funs, u_guess, u_error, pixel_grid):
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

    # initialize a value for theta
    return lnpdf, sample_from_prior #star_color_mog


if __name__=="__main__":

    #########################################
    # set true parameters to be inferred    #
    #########################################
    u = sc.img_constants['r']['phi'] + 1./3600.  # near center, off by a pixel
    flux_dict = {'g': 9.6173432087297002,
                 'i': 33.070941854638555,
                 'r': 24.437380835296388,
                 'u': 1.2582444245272928,
                 'z': 40.854689375715807}
    eps_dict  = {'u': 28., 'g': 307., 'r': 684., 'i': 817, 'z': 484.}
    true_colors = fluxes_to_colors(np.array([flux_dict[b] for b in bands]))
    true_params = np.concatenate([true_colors, u])

    ###################################################################
    # create model image generating functions, specific to the image  #
    # described in the image_constants module                         #
    ###################################################################
    img_shape = (50, 50)
    img_rho   = np.array([25., 25.])
    img_funs  = { k: make_gen_model_image(
                            phi              = ic['phi'],
                            rho              = img_rho,
                            Ups_inv          = ic['Ups_inv'],
                            psf              = ic['psf'],
                            band             = ic['band'],
                            photons_per_nmgy = ic['photons_per_nmgy'])
                       for k, ic in sc.img_constants.iteritems() }
    model_img_funs  = { k: img_funs[k][0] for k in bands }
    equa2pixel_funs = { k: img_funs[k][1] for k in bands }
    pixel2equa_funs = { k: img_funs[k][2] for k in bands }


    ######################################
    # generate synthetic image patches   #
    ######################################
    shape = (50, 50)
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    pixel_grid = np.column_stack([xx.flatten(), yy.flatten()])
    imgdict = { b: sample_image(true_params, model_img_funs[b],
                                eps_dict[b], pixel_grid).reshape(xx.shape)
                for b in bands }

    # plot
    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns; sns.set_style("white")
    plt.matshow(imgdict['r'])

    #####################################################
    # create logpdf function handle (and gradient)      #
    #####################################################
    # first find the brightest pixel location
    ubar, ustd = get_brightest_location(imgdict, pixel2equa_funs)

    # construct lnpdf and prior sample fun
    lnpdf, sample_from_prior = make_lnpdf_fun(imgdict, eps_dict,
                                              model_img_funs = model_img_funs,
                                              u_guess = ubar,
                                              u_error = ustd,
                                              pixel_grid = pixel_grid)
    glnpdf = grad(lnpdf)

    #####################################################
    # find "map" (or best approx) and curvature at map  #
    #####################################################
    from scipy.optimize import minimize
    res = minimize(fun     = lambda th: -1.*lnpdf(th),
                   jac     = lambda th: -1.*grad(lnpdf)(th),
                   x0      = sample_from_prior(),
                   method  ='L-BFGS-B',
                   options = {'maxiter':100, 'disp':10, 'ftol':1e-20})
    th_map = res.x.copy()
    H_map  = hessian(lnpdf)(th_map)
    Sig    = np.linalg.inv(-H_map)
    sig2   = np.diag(Sig)

    print "lnpdf value at map         ", lnpdf(th_map)
    print "lnpdf value at true params ", lnpdf(true_params)

    # at map, plot out random direction LNPDF values
    def eval_random_dir(th, fun, zmin=-5, zmax=5, num_z=100):
        rdir = np.random.randn(th.shape[0])
        rdir /= np.sqrt(np.sum(rdir*rdir))
        lnpdfz = lambda z: lnpdf(rdir*z + th)
        zgrid = np.linspace(zmin, zmax, num_z)
        return np.array([lnpdfz(z) for z in zgrid])

    fig = plt.figure(figsize=(12, 6))
    for _ in xrange(5):
        llz = eval_random_dir(th_map, lnpdf)
        plt.plot(llz)


    ##################################################################
    # mcmc with MH - construct metropolis hastings sample functions, #
    # and run multiple chains                                        #
    ##################################################################
    from CelestePy.util.infer.mcmc import mcmc_multi_chain
    from CelestePy.util.infer.mh import mhstep
    from CelestePy.util.infer.slicesample import slicesample
    Nchains = 8
    th0s      = 1.1 * np.array([sample_from_prior() for _ in xrange(Nchains)])
    ll0s      = np.array([ lnpdf(th) for th in th0s ])
    #mcmc_funs = [lambda th, llth: mhstep(th, lnpdf, llx=llth, prop_sig2=.15*sig2)
    #             for _ in xrange(Nchains)]
    mcmc_funs = [lambda th, llth: slicesample(th, lnpdf, compwise=True)
                 for _ in xrange(Nchains)]
    th_samps, ll_samps = \
        mcmc_multi_chain(th0s, ll0s, mcmc_funs, Nsamps=100, burnin=50)

    def plot_chain_marginals(th_samps, true_params):
        Nchains, Nsamps, D = th_samps.shape
        plot_colors = sns.color_palette(n_colors=Nchains)
        fig, axarr  = plt.subplots(2, D/2 + 1, figsize=(12,8))
        for d, ax in enumerate(axarr.flatten()[:D]):
            ths = th_samps[:,:,d]
            for k in xrange(Nchains):
                c = plot_colors[k]
                ax.hist(ths[k,Nsamps/2:], alpha=.2, color=c)
            ax.scatter(true_params[d], 0, s=10, marker='x', color='red')
        return fig, axarr

    def plot_pairwise(th_samps, true_params):
        import pandas as pd
        samp_df = pd.DataFrame(np.hstack([th_samps[len(th_samps)/2:,:5],
                                          100*th_samps[len(th_samps)/2:, 5:7]]),
                               columns=['lnr', 'cu', 'cg', 'cr', 'ci', 'ra', 'dec'])
        pplot = sns.pairplot(samp_df, size=1.5)
        tp_scaled = true_params.copy()
        tp_scaled[5:7] *= 100.
        for tp, ax in zip(tp_scaled, pplot.diag_axes):
            ax.scatter(tp, 0, c='red', marker='x', s=50)

    plot_chain_marginals(th_samps, true_params)


    ##################################################################
    # MCMC with parallel tempering - construct multiple parallel     #
    # tempering chains (using MH within) compare mixing              #
    ##################################################################
    import CelestePy.util.infer.parallel_tempering as pt

    # create parrallel tempering step functions
    def make_pt_step():
        num_temps = 12
        def mh_step_maker(lnpdf):
            #return lambda th, llth: mhstep(th, lnpdf, llx=llth, prop_sig2=.1*sig2)
            return lambda th, llth: slicesample(th, lnpdf, compwise=False)

        temp_th0s = np.array([sample_from_prior() for _ in xrange(num_temps)])
        pt_step, pt_swaps = pt.make_parallel_tempering_sample_fun(
                                    th0s     = temp_th0s,
                                    lnpdf    = lnpdf,
                                    invtemps = np.linspace(.01, 1., num_temps),
                                    mcmc_step_maker = mh_step_maker)
        return pt_step, pt_swaps

    pt_steps_and_swaps = [make_pt_step() for _ in range(Nchains)]
    pt_steps           = [p[0] for p in pt_steps_and_swaps]
    pt_swaps           = [p[1] for p in pt_steps_and_swaps]
    th0s = np.array([1.1*sample_from_prior() for _ in range(Nchains)])
    ll0s = np.array([lnpdf(th) for th in th0s])

    def callback(n):
        print "swaps in chains: "
        for pts in pt_swaps:
            print pts()

    th_samps, ll_samps = \
        mcmc_multi_chain(th0s, ll0s, pt_steps, Nsamps=200, burnin=100, callback=callback)


    plot_chain_marginals(th_samps, true_params)


    plot_pairwise(th_samps[0], true_params)


    sys.exit

    # TODO plot the LL contribution from each pixel (over time) as the chain moves
    # turn into a movie
    ll_imgs = []
    for params in th_samps[::100]:
        imgi = np.zeros((len(bands),) + xx.shape)
        for bi, b in enumerate(bands):
            mimg = model_img_funs[b](params, pixel_grid)
            llimg = poisson_logpmf(imgdict[b].ravel(), mimg).reshape(xx.shape)
            imgi[bi,:,:] = llimg
        ll_imgs.append(imgi)

    fig, axarr = plt.subplots(2, 3, figsize=(8,6))
    idx = 0
    for bi, (ax, b) in enumerate(zip(axarr.flatten(), bands)):
        ax.matshow(ll_imgs[idx][bi,:,:])


    print true_params
    print np.mean(th_samps, 0)
    print np.percentile(th_samps, [2.5, 97.5], axis=0)
#
#    # plot fits compared to truth
#    burn_in = 1500
#    fig, axarr = plt.subplots(2, 3, figsize=(18, 4))
#    for i, ax in enumerate(axarr.flatten()[:5]):
#        n, _, _ = ax.hist(th_samps[burn_in:, i+2], bins=25, normed=True, alpha=.25)
#        ax.plot([th_true[i+2], th_true[i+2]], [0., np.max(n)])
#

    # multiple chains - track r hat - different 



    #param_samps
    # plot posterior marginals


#import pandas as pd
#import pyprind, os
#import cPickle as pickle
#import CelestePy.util.data as data_util
#import CelestePy.models.gmm_prior as models
#from CelestePy.util.data.io import celestedf_row_to_params
#from CelestePy.model_base import generate_background_patch
#
#if __name__ == '__main__':
#
#    ########################################################
#    # load cached stripe 82 data source
#    ########################################################
#    data_dict          = pickle.load(open('s82_dict.pkl', 'rb'))
#    run, camcol, field = data_dict['run'], data_dict['camcol'], data_dict['field']
#    primary_field_df   = data_dict['primary_field_df']
#    coadd_field_df     = data_dict['coadd_field_df']
#
#    # load in fits images
#    imgfits = data_util.make_fits_images(run, camcol, field)
#    for k, img in imgfits.iteritems():
#        img.epsilon = np.median(img.nelec)
#
#    ######################################################
#    # select a source, fit celeste to it
#    ######################################################
#    reload(models)
#
#    # initialize celeste model, add images
#    model = models.CelesteGMMPrior(images=imgfits.values())
#    model.add_field(img_dict = imgfits)
#
#    # initializes each source as the photo obj says it was
#    model.initialize_sources(init_srcs = [model.init_celeste_source_from_df(r)
#                                          for _, r in primary_field_df.iterrows()])
#
#    ######################################################################
#    # select star, make background image
#    ######################################################################
#    idx = 0
#    src = model.srcs[idx]
#    src.background_image_dict = {img: generate_background_patch(src, model.srcs, img)
#                                 for img in imgfits.values()}
#    for k, v in src.background_image_dict.iteritems():
#        v[:] = 100.
#
#    img_dict, mask_dict = sample_synthetic_data(src, imgfits)
#
#    fig, axarr = plt.subplots(1, 3, figsize=(12, 6))
#    src.plot(imgfits['r'], *axarr)
#
#    src_row       = primary_field_df.iloc[idx]
#    src_row_coadd = coadd_field_df.iloc[idx]
#
#    #####################################################################
#    # create likelihood function handle, draw MH samples
#    #####################################################################
#    star_logp, dstar_logp, th_star, unpack_star_params, pack_star_params = \
#        src.make_star_unconstrained_logp(data_img_dict = img_dict,
#                                         mask_img_dict = mask_dict,
#                                         background_img_dict = src.background_image_dict)
#    th_true = th_star.copy()
#

#    ax = axarr.flatten()[-1]
#    u_samps = src.constrain_loc(th_samps[burn_in:,:2])
#    u_lo, u_hi = np.min(u_samps, axis=0), np.max(u_samps, axis=0)
#    ax.scatter(u_samps[::10,0], u_samps[::10,1], s=30, c='grey')
#    ax.set_xlim([u_lo[0], u_hi[0]])
#    ax.set_ylim([u_lo[1], u_hi[1]])
#    ax.scatter(src.params.u[0], src.params.u[1], s=100, c='red')
#
#    fig, ax = plt.subplots(1, 1, figsize=(12,8))
#    ax.plot(ll_samps[burn_in:])
#
#
