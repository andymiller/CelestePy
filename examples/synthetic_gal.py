"""
Various MCMC techniques to infer star parameters.  Model and functions are
described in CelestePy.util.point_source
"""
import matplotlib.pyplot as plt; plt.ion()
import seaborn as sns; sns.set_style("white")
import autograd.numpy as np
from autograd import grad, hessian
from CelestePy.galaxy import make_gen_model_image, \
                             make_lnpdf_fun, bands
import synthetic_constants as sc
from CelestePy.util.transform import fluxes_to_colors, unconstrain_gal_shape, \
                                     constrain_gal_shape


def plot_chain_marginals(th_samps, true_params, names):
    Nchains, Nsamps, D = th_samps.shape
    plot_colors = sns.color_palette(n_colors=Nchains)
    fig, axarr  = plt.subplots(2, D/2 + 1, figsize=(12,8))
    for d, ax in enumerate(axarr.flatten()[:D]):
        ths = th_samps[:,:,d]
        for k in xrange(Nchains):
            c = plot_colors[k]
            ax.hist(ths[k,Nsamps/2:], alpha=.2, color=c, normed=True)
        ax.scatter(true_params[d], 0, s=50, marker='x', color='red', linewidth=10)
        ax.set_ylim(bottom=0.)
        ax.set_title(names[d])
    fig.tight_layout()
    return fig, axarr


#def plot_pairwise(th_samps, true_params):
#    import pandas as pd
#    samp_df = pd.DataFrame(np.hstack([th_samps[len(th_samps)/2:,:5],
#                                      100*th_samps[len(th_samps)/2:, 5:7]]),
#                           columns=['lnr', 'cu', 'cg', 'cr', 'ci', 'ra', 'dec'])
#    pplot = sns.pairplot(samp_df, size=1.5)
#    tp_scaled = true_params.copy()
#    tp_scaled[5:7] *= 100.
#    for tp, ax in zip(tp_scaled, pplot.diag_axes):
#        ax.scatter(tp, 0, c='red', marker='x', s=50)
#

if __name__=="__main__":

    #########################################
    # set true parameters to be inferred    #
    #########################################
    true_u = sc.img_constants['r']['phi'] + 1./3600.  # near center, off by a pixel
    flux_dict = {'g': 9.6173432087297002,
                 'i': 33.070941854638555,
                 'r': 24.437380835296388,
                 'u': 1.2582444245272928,
                 'z': 40.854689375715807}
    eps_dict  = {'u': 28., 'g': 307., 'r': 684., 'i': 817, 'z': 484.}
    eps_dict  = {b: eps_dict[b] / 100. for b in bands}
    true_shape= np.array([ .9, # theta_s
                           40., # sig2_s (in pixels)
                           90., # phi, north rotation [0, 180]
                            .1  # rho min / major axis ratio
                         ])
    true_colors = np.array([2.80342137, -1.00360455, -.59958647, -0.47531305,  0.34646993])
    true_colors = 2*np.array([3., -.5, .5, .5, 1.])
    #true_colors = fluxes_to_colors(np.array([flux_dict[b] for b in bands]))
    true_params = np.concatenate([true_colors, true_u,
                                  unconstrain_gal_shape(true_shape)])

    ###################################################################
    # create model image generating functions, specific to the image  #
    # described in the image_constants module                         #
    ###################################################################
    img_shape = (50, 50)
    img_rho   = np.array([25., 25.])
    img_phi   = sc.img_constants['r']['phi'] # make all images have the same reference RA/DEC

    # make image functions
    # returns a tuple of functions: gen_psf_image, get_brightest_radec,
    #                               sample_image
    #                               gen_psf_image_fixed_location_maker
    img_funs  = { k: make_gen_model_image(
                            phi              = img_phi, # ic['phi']
                            rho              = img_rho,
                            Ups_inv          = ic['Ups_inv'],
                            psf              = ic['psf'],
                            band             = ic['band'],
                            photons_per_nmgy = ic['photons_per_nmgy'])
                       for k, ic in sc.img_constants.iteritems() }
    model_img_funs       = { k: img_funs[k][0] for k in bands }
    brightest_funs       = { k: img_funs[k][1] for k in bands }
    sample_fun           = { k: img_funs[k][2] for k in bands }
    model_img_fixed_funs = { k: img_funs[k][3] for k in bands }
    unit_flux_funs       = { k: img_funs[k][4] for k in bands }

    ######################################
    # generate synthetic image patches   #
    ######################################
    shape = (50, 50)
    xg = np.arange(shape[0])
    yg = np.arange(shape[1])
    xx, yy = np.meshgrid(xg, yg)
    pixel_grid = np.ascontiguousarray(np.column_stack([xx.flatten(), yy.flatten()]))
    imgdict = { b: sample_fun[b](true_params, eps_dict[b],
                                 pixel_grid).reshape(xx.shape)
                for b in bands }

    # plot observations
    from CelestePy.util.plots import add_colorbar_to_axis
    fig, axarr = plt.subplots(1, 5, figsize=(16,4))
    for b, ax in zip(bands, axarr.flatten()):
        cim = ax.imshow(imgdict[b], interpolation='none')
        add_colorbar_to_axis(ax, cim)
        ax.set_title("band %s"%b)
    fig.tight_layout()
    sys.exito

    #####################################################
    # create logpdf function handle (and gradient)      #
    #####################################################
    # first find the brightest pixel location
    us = np.array([brightest_funs[b](imgdict[b]) for b in bands])
    ubar, ustd = us.mean(0), us.std(0)

    # create sub pixel grid
    #xxs, yys = np.meshgrid(xg[::2], yg[::2])
    # sub_pixel_grid = np.meshgrid(xxs.flatten(), yys.flatten())

    # construct lnpdf and prior sample fun
    lnpdf, sample_from_prior, lnpdf_u_maker = \
        make_lnpdf_fun(imgdict, eps_dict,
                       model_img_funs = model_img_funs,
                       u_guess = ubar,
                       u_error = ustd,
                       pixel_grid = pixel_grid, 
                       psf_image_fixed_location_makers = model_img_fixed_funs)
    print " made lnpdf - eval at true = ", lnpdf(true_params)
    #glnpdf = grad(lnpdf)

    #from CelestePy.util.misc import eval_random_dir
    #import pyprind
    #sig2grid = np.linspace(.01, 100, 100)
    #eparams = np.zeros((len(sig2grid), true_params.shape[0]))
    #th_map_shp = constrain_gal_shape(th_map[-4:])
    #for i in pyprind.prog_bar(xrange(len(sig2grid))):
    #    shp = th_map_shp.copy() #shape.copy()
    #    shp[2] = sig2grid[i]
    #    eparams[i] = np.concatenate([th_map[:-4],
    #                                 unconstrain_gal_shape(shp)])
    #lls = np.array([lnpdf(e) for e in eparams])
    #plt.plot(sig2grid, np.exp(lls-lls.max()))

    #####################################################
    # find "map" (or best approx) and curvature at map  #
    #####################################################
    #from scipy.optimize import minimize
    #res = minimize(fun     = lambda th: -1.*lnpdf(th),
    #               #jac     = lambda th: -1.*grad(lnpdf)(th),
    #               x0      = sample_from_prior(),
    #               method  ='Nelder-Mead',
    #               options = {'maxiter':10, 'disp':10, 'ftol':1e-10})
    #th_map = res.x.copy()
    #colors_map, u_map, shape_map = th_map[:5], th_map[5:7], th_map[7:]
    #print "Map : th, sig, phi, rho", constrain_gal_shape(shape_map)
    #print "true: th, sig, phi, rho", true_shape
    #H_map  = hessian(lnpdf)(th_map)
    #Sig    = np.linalg.inv(-H_map)
    #sig2   = np.diag(Sig)
    #print "lnpdf value at map         ", lnpdf(th_map)
    #print "lnpdf value at true params ", lnpdf(true_params)
    # at map, plot out random direction LNPDF values
    #from CelestePy.util.misc import eval_random_dir
    #fig = plt.figure(figsize=(12, 6))
    #for _ in xrange(2):
    #    llz = eval_random_dir(th_map, lnpdf)
    #    plt.plot(llz)


    ######################################################################
    # mcmc with MH/slice sampling - construct metropolis hastings sample #
    # functions, and run multiple chains                                 #
    ######################################################################
    from CelestePy.util.infer.mcmc import mcmc_multi_chain
    from CelestePy.util.infer.mh import mhstep
    from CelestePy.util.infer.slicesample import slicesample
    Nchains = 8
    th0s    = 1.1 * np.array([sample_from_prior() for _ in xrange(Nchains)])
    ll0s    = np.array([ lnpdf(th) for th in th0s ])

    def gibbs_step(params, llth):
        """ steps between colors, u, shape """
        colors, u, shape = params[0:5], params[5:7], params[7:]
        lnpdfu = lnpdf_u_maker(u, shape)
        for _ in range(5):
            colors, _ = slicesample(colors, lnpdfu, compise=True)
        lnpdf_loc = lambda u: lnpdf(np.concatenate([colors, u, shape]))
        u, ll     = slicesample(u, lnpdf_loc, compwise=False)
        lnpdf_shp = lambda s: lnpdf(np.concatenate([colors, u, s]))
        shape, ll = slicesample(shape, lnpdf_shp, compwise=True)
        return np.concatenate([colors, u, shape]), ll

    gibbs_funs = [gibbs_step for _ in xrange(Nchains)]
    np.random.seed(42)
    print " gibbs step ..."
    gibbs_step(true_params, lnpdf(true_params))
    print " ... test done"

    th_samps, ll_samps = \
        mcmc_multi_chain(th0s, ll0s, gibbs_funs, Nsamps=500, burnin=100, n_jobs=8)
    _, Nsamps, D = th_samps.shape

    import cPickle as pickle
    with open('synthetic_gal_samps_3.pkl', 'wb') as f:
        pickle.dump(th_samps, f)
        pickle.dump(ll_samps, f)

    sys.exit

    with open('synthetic_gal_samps_4.pkl', 'rb') as f:
        th_samps = pickle.load(f)
        ll_samps = pickle.load(f)
        Nchains, Nsamps, D = th_samps.shape

    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns; sns.set_style("white")
    fig, axarr = plt.subplots(3, 1, figsize=(12,4))
    axarr[0].plot(ll_samps[:, Nsamps/2:].T)
    axarr[0].set_title("log likelihood")
    axarr[1].plot(th_samps[:, Nsamps/2:, 0].T)
    axarr[1].set_title("ln r trace")
    axarr[2].plot(th_samps[:, Nsamps/2:, -2].T)
    axarr[2].set_title("ra trace")

    th_shape = np.array([
        np.array([constrain_gal_shape(th[-4:]) for th in th_samps[k,:,:]])
        for k in xrange(th_samps.shape[0])])
    plot_chain_marginals(th_shape[:,Nsamps/2:,:], true_shape, names=['th', 'sig2', 'angle', 'ab'])
    print th_shape.mean(axis=(0,1))
    print true_shape 

    import CelestePy.util.infer.mcmc_diagnostics as mcd
    mcd.compute_n_eff_acf(th_shape[0,:,0])

    th_flat = np.row_stack([ th_samps[k][Nsamps/2:,:] for k in xrange(th_samps.shape[0]) ])
    names = ['lnr', 'cu', 'cg', 'cr', 'ci', 'ra', 'dec', 'th', 'sig2', 'angle', 'ab']
    plot_chain_marginals(th_samps[:,Nsamps/2:,:], true_params, names)
    plot_pairwise(th_flat, true_params)



