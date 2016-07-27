import autograd.numpy as np
import autograd.numpy.random as npr
from mcmc import mcmc_multi_chain


def make_parallel_tempering_sample_fun(th0s, lnpdf, invtemps, mcmc_step_maker):
    """
    returns a function handle that does one parallel tempering move
    Args:
        th0s           : num_temps x D np array.  starting location for all chains
        lnpdf          : target logpdf for the sampler (at the coolest temp)
        invtemps       : inverse temperature schedule
        mcmc_step_maker: function that returns an MCMC step function, e.g.
                            hot_mcmc_step = mcmc_step_maker(hot_lnpdf)
                            new_x, new_ll = hot_mcmc_step(x, ll)
                         where the stepper runs one step of a markov chain that
                         leaves the input distribution, hot_lnpdf, invariant. 
                         The details of this are left to the mcmc_step_maker
                         function.
    """
    num_temps, D = th0s.shape
    assert len(invtemps) == num_temps
    assert invtemps[-1] == 1.

    # make hot chains
    hot_lnpdfs  = [lambda th, t=t: t * lnpdf(th) for t in invtemps]

    # make mcmc steppers
    mcmc_steps = [mcmc_step_maker(hlnpdf) for hlnpdf in hot_lnpdfs]

    # create current state of the chain and current loglikes
    curr_th = th0s.copy()
    curr_ll = np.array([hlnpdf(th) for hlnpdf, th in zip(hot_lnpdfs, curr_th)])

    # create the parallel tempering sample function
    num_swaps = np.zeros(num_temps)
    def pt_samplefun(th, ll):
        curr_th[-1], curr_ll[-1] = th, ll
        for k, mcmc_step in enumerate(mcmc_steps):
            curr_th[k], curr_ll[k] = mcmc_step(curr_th[k], curr_ll[k])

        # propose swaps cascading down from first 
        for ci in range(num_temps-1):
            # cache raw ll's for each (already computed)
            ll_ci      = curr_ll[ci]   / invtemps[ci]      # ln pi(th_ci)
            ll_ci_plus = curr_ll[ci+1] / invtemps[ci+1]    # ln pi(th_ci+1)

            # propose swap between chain index ci and ci + 1
            ll_prop = invtemps[ci]*ll_ci_plus + invtemps[ci+1]*ll_ci
            ll_curr = invtemps[ci]*ll_ci      + invtemps[ci+1]*ll_ci_plus
            if np.log(npr.rand()) < ll_prop - ll_curr:

                # swap th_ci with th_ci+1 (and ll's)
                th_ci         = curr_th[ci]
                curr_th[ci]   = curr_th[ci+1]
                curr_th[ci+1] = th_ci

                ll_ci         = curr_ll[ci]
                curr_ll[ci]   = curr_ll[ci+1]
                curr_ll[ci+1] = ll_ci

                # track swaps
                num_swaps[ci] += 1

        return curr_th[-1], curr_ll[-1]

    num_swaps_fun = lambda: num_swaps
    return pt_samplefun, num_swaps_fun


from CelestePy.util.infer.slicesample import slicesample
def parallel_temper_slice(lnpdf, x0, Nsamps, Nchains, temps=None,
                          callback=None, verbose=True, printskip=20,
                          compwise = True):
    if temps is None:
        temps = np.linspace(.01, 1., Nchains)

    def slice_step_maker(hlnpdf):
        return lambda th, llth: slicesample(th, hlnpdf, compwise=False)

    pt_step, pt_waps = \
        make_parallel_tempering_sample_fun(th0s     = x0.copy(),
                                           lnpdf    = lnpdf,
                                           invtemps = temps,
                                           mcmc_step_maker = slice_step_maker)

    th0 = np.array([ x0[0] ])
    ll0 = np.array([ lnpdf(th) for th in th0 ])
    th_samps, ll_samps = \
        mcmc_multi_chain(th0, ll0, [pt_step], Nsamps=Nsamps, burnin=Nsamps/2,
                         callback=callback)

    return th_samps[0], ll_samps[0]


if __name__=="__main__":
    import numpy.linalg as npla
    import scipy.misc as scpm
    from CelestePy.util.dists.mog import mog_loglike, mog_samples

    # Create a random parameterization.
    def gen_mog_2d(seed=105, K=3, alpha=1.):
        rng   = npr.RandomState(seed)
        means = rng.randn(K,2)
        covs  = rng.randn(K,2,2)
        covs  = np.einsum('...ij,...kj->...ik', covs, covs)
        icovs = np.array([npla.inv(cov) for cov in covs])
        dets  = np.array([npla.det(cov) for cov in covs])
        chols = np.array([npla.cholesky(cov) for cov in covs])
        pis   = rng.dirichlet(alpha*np.ones(K))
        return means, covs, icovs, dets, chols, pis

    means, covs, icovs, dets, chols, pis = gen_mog_2d()
    mu    = np.sum(pis * means.T, axis=1)
    lnpdf = lambda x: mog_loglike(x, means, icovs, dets, pis)

    #test on simple two d example
    Nchains = 5
    Nsamps  = 5000
    x0      = np.random.randn(Nchains, 2)
    chain, chain_lls = parallel_temper_slice(
        lnpdf   = lnpdf,
        x0      = np.random.randn(Nchains, 2),
        Nsamps  = Nsamps,
        Nchains = Nchains,
        verbose = True, printskip=100)

    sstep = lambda th, ll: slicesample(th, lnpdf, compwise=False)
    th0 = np.array([ x0[0] ])
    ll0 = np.array([ lnpdf(x0[0]) ])
    slice_chain, slice_lls = \
        mcmc_multi_chain(th0, ll0, [sstep], Nsamps=Nsamps, burnin=Nsamps/2)
    slice_chain, slice_lls = slice_chain[0], slice_lls[0]

    # perfect samples for comparison
    S = mog_samples(Nsamps, means, chols, pis)

    # diagnostics
    print "true     mean : ", np.str(mu)
    print "pt slice mean : ", np.str(chain.mean(axis=0))
    print "slice    mean : ", np.str(slice_chain.mean(axis=0))

    import matplotlib.pyplot as plt; plt.ion()
    import seaborn as sns; sns.set_style('white')
    Sm = np.cumsum(S, axis=0) / (np.arange(Nsamps)[:,np.newaxis]+1.0)
    Xm = np.cumsum(chain, axis=0) / (np.arange(Nsamps)[:,np.newaxis]+1.0)
    SSm = np.cumsum(slice_chain, axis=0) / (np.arange(Nsamps)[:,np.newaxis]+1.0)
    plt.figure(1)
    plt.plot(np.arange(Nsamps), Sm, 'k-',
             np.arange(Nsamps), Xm, 'b-',
             np.arange(Nsamps), SSm, 'r-')
    plt.show()

    import mcmc_diagnostics as mcd
    mcd.compute_n_eff_acf(chain[:,0])
    mcd.compute_n_eff_acf(slice_chain[:,0])


#def parallel_temper_slice(lnpdf, x0, Nsamps, Nchains, temps=None, 
#                          callback=None, verbose=True, printskip=20, 
#                          compwise = True):
#
#    # verify temperature schedule
#    if temps is None:
#        temps      = np.linspace(.2, 1., Nchains)
#    assert len(temps) == Nchains, "Nchains must be = len(temps)!"
#    assert temps[-1] == 1., "temps[-1] must be = 1. or else you're not doing it right..."
#
#    # keep track of swaps into every temperature state
#    Nswaps = np.zeros(len(temps)) 
#    print " === parallel tempering ==="
#    print "  with temps ", np.str(temps)
#
#    # set up printing
#    def printif(string, condition):
#        if condition:
#            print string
#    printif("{iter:10}|{ll:10}|{num_swaps:10}|{cold_swaps:12}|{th0:10}".format(
#                iter       = " iter ",
#                ll         = " ln_post ", 
#                num_swaps  = " Nswaps ",
#                cold_swaps = " NColdSwaps ",
#                th0        = " th0 (mean, sd)"), verbose)
#
#    # set up sample array
#    assert x0.shape[0] == Nchains, "initial x has to have shape Nchains x Dim"
#    D = x0.shape[1]
#    chain        = np.zeros((Nchains, Nsamps, D))
#    chain[:,0,:] = x0.copy()
#    chain_lls    = np.zeros((Nchains, Nsamps))
#    for ci in xrange(Nchains):
#        chain_lls[ci, 0] = temps[ci] * lnpdf(chain[ci, 0, :])
#
#    # draw samples
#    for s in np.arange(1, Nsamps):
#
#        # Nchains HMC draws
#        for ci in range(Nchains):
#            chain[ci, s, :], chain_lls[ci, s] = slicesample(
#                    init_x   = chain[ci][s-1,:],
#                    logprob  = lambda(x): temps[ci] * lnpdf(x),
#                    compwise = compwise
#                )
#
#        # propose swaps cascading down from first 
#        for ci in range(Nchains-1):
#            # cache raw ll's for each (already computed)
#            ll_ci      = chain_lls[ci][s] / temps[ci]
#            ll_ci_plus = chain_lls[ci+1][s] / temps[ci + 1]
#
#            # propose swap between chain index ci and ci + 1
#            ll_prop = temps[ci]*ll_ci_plus + temps[ci+1]*ll_ci
#            ll_curr = chain_lls[ci][s] + chain_lls[ci+1][s]
#            if np.log(npr.rand()) < ll_prop - ll_curr:
#                ci_samp = chain[ci, s, :].copy()
#
#                # move chain sample ci+1 into ci
#                chain[ci, s, :]   = chain[ci+1, s, :]
#                chain_lls[ci, s]  = temps[ci] * ll_ci_plus
#
#                # move chain sample ci into ci + 1
#                chain[ci+1, s, :]  = ci_samp
#                chain_lls[ci+1, s] = temps[ci+1]*ll_ci
#
#                # track number of swaps
#                Nswaps[ci+1] += 1
#
#        printif("{iter:10}|{ll:10}|{num_swaps:10}|{cold_swaps:12}|{th0:10}".format(
#                iter       = "%d/%d"%(s, Nsamps),
#                ll         = " %2.4g "%chain_lls[-1, s],
#                num_swaps  = " %d "%np.sum(Nswaps),
#                cold_swaps = " %d"%Nswaps[-1],
#                th0        = " %2.2f (%2.2f, %2.2f)"%(chain[-1,s,0], chain[-1,:s,0].mean(), chain[-1,:s,0].std())), 
#                verbose and s%printskip==0)
#
#        if callback is not None:
#            callback(s, chain, chain_lls)
#
#    #only return the chain we care about
#    return chain, chain_lls


