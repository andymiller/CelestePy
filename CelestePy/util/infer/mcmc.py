"""
General functions for simulating multiple MCMC chains
"""
import autograd.numpy as np
import mcmc_diagnostics as mcd
import pyprind
from joblib import Parallel, delayed

def mcstep(th, ll, sfun):
    return sfun(th, ll)

def mcmc_multi_chain(th0s, ll0s, sample_funs, Nsamps, burnin=100,
                     callback=None, prog_bar=False, n_jobs=None):
    import CelestePy.util.infer.mcmc_diagnostics as mcd
    _, D     = th0s.shape
    Nchains  = len(sample_funs)
    assert th0s.shape[0] == Nchains

    # allocate chains and initialize to starting values
    th_samps = np.zeros((Nchains, Nsamps, D))
    ll_samps = np.zeros((Nchains, Nsamps))
    for k in xrange(Nchains):
        th_samps[k,0,:] = th0s[k]
        ll_samps[k,0]   = ll0s[k]

    # run each chain forward Nsamps steps
    for n in xrange(1, Nsamps):

        # serial (no num_jobs specified)
        if n_jobs is None:
            for k, sfun in enumerate(sample_funs):
                th_samps[k,n,:], ll_samps[k,n] = \
                    sfun(th_samps[k,n-1,:], ll_samps[k,n-1])
        else:
            th_ll_list = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(mcstep)(th_samps[k,n-1,:], ll_samps[k,n-1], sample_funs[k])
                                for k in xrange(Nchains)
                )
            for k, (th, ll) in enumerate(th_ll_list):
                th_samps[k,n,:], ll_samps[k,n] = th, ll

        # report r hat, and other sampler statistics
        if Nchains > 1:
            if n > 2:
                if n == burnin+2:
                    print "... done burning in"
                start = burnin if n > burnin+2 else 0
                rhats = np.array([mcd.compute_r_hat(th_samps[:,start:n,d])
                                  for d in xrange(D)])
                print "iter %d"%n, ["%2.2f"%r for r in rhats]
        else:
            if n % 100 == 0:
                print "iter %d: ll = %2.3f" % (n, ll_samps[0,n])

        if callback is not None:
            callback(n)

    return th_samps, ll_samps


