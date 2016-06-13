import autograd.numpy as np
import autograd.numpy.random as npr

def mhstep(x, logp, llx=None, prop_sig2 = .1):
    """ run one metropolis step forward with a symmetric proposal
    Args:
        x         : current state of the chain
        logp      : log probability density of equilibrium distribution
        llx       : (optional) llx = logp(x)
        prop_sig2 : gaussian variance of proposal
    """
    # cache current loglikelhood
    if llx is None:
        llx = logp(x)
    # propose
    prop    = np.sqrt(prop_sig2) * npr.randn(len(x)) + x
    ll_prop = logp(prop)
    # accept-reject
    if np.log(npr.rand()) < (ll_prop - llx):
        x   = prop
        llx = ll_prop
    return x, llx

