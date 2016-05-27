"""
Customized Source and Celeste Model derived class

Star and Galaxy fluxes governed by a mixture of Gaussians
"""
from CelestePy.model_base import CelesteBase
from CelestePy.source_base import Source
from CelestePy.source_params import SrcParams
import autograd.numpy as np
from autograd import grad
import cPickle as pickle
from scipy.stats import multivariate_normal as mvn
import CelestePy.util.data as du
from CelestePy.util.infer.hmc import hmc

#############################
#source parameter priors    #
#############################
import os
prior_param_dir = os.path.join(os.path.dirname(__file__),
                               '../experiments/empirical_priors')
prior_param_dir = '../empirical_priors/'
from os.path import join
star_flux_mog = pickle.load(open(join(prior_param_dir, 'star_fluxes_mog.pkl'), 'rb'))
gal_flux_mog  = pickle.load(open(join(prior_param_dir, 'gal_fluxes_mog.pkl'), 'rb'))
gal_re_mog    = pickle.load(open(join(prior_param_dir, 'gal_re_mog.pkl'), 'rb'))
gal_ab_mog    = pickle.load(open(join(prior_param_dir, 'gal_ab_mog.pkl'), 'rb'))
star_mag_proposal = pickle.load(open(join(prior_param_dir, 'star_mag_proposal.pkl'), 'rb'))
gal_mag_proposal  = pickle.load(open(join(prior_param_dir, 'gal_mag_proposal.pkl'), 'rb'))
star_rad_proposal = pickle.load(open(join(prior_param_dir, 'star_res_proposal.pkl'), 'rb'))

# inflate covariance
#gal_mag_proposal.res_covariance *= .001
#star_mag_proposal.res_covariance *= .001

def contains(pt, lower, upper):
    return np.all( (pt > lower) & (pt < upper) )


############################################################
# subclass source base class - implement resample method   #
############################################################

class SourceGMMPrior(Source):
    """Source with a gaussian mixture model prior over the source parameters"""
    def __init__(self, params, model, imgs):
        super(SourceGMMPrior, self).__init__(params, model, imgs)

    def location_logprior(self, u):
        if contains(u, self.u_lower, self.u_upper):
            return 0.
        else:
            return -np.inf

    def resample(self):
        #assert len(self.sample_image_list) != 0, "resample source needs sampled source images"
        assert len(self.background_image_dict) != 0, "need background images to sample"
        if self.is_star():
            self.resample_star()
        elif self.is_galaxy():
            self.resample_galaxy()

    def constrain_loc(self, u_unc):
        u_unit = 1./(1. + np.exp(-u_unc))
        return u_unit * (self.u_upper - self.u_lower) + self.u_lower

    def unconstrain_loc(self, u):
        assert contains(u, self.u_lower, self.u_upper), "point not contained in initial interval!"
        # convert to unit interval, and then apply logit transformation
        u_unit = (u - self.u_lower) / (self.u_upper - self.u_lower)
        return np.log(u_unit) - np.log(1. - u_unit)

    def constrain_shape(self, lg_shape):
        lg_theta, lg_sigma, lg_phi, lg_rho = lg_shape
        theta = 1./(1. + np.exp(-lg_theta))
        sigma = np.exp(lg_sigma)
        phi   = 1./(1. + np.exp(-lg_phi)) * (180) + -180
        rho   = 1./(1. + np.exp(-lg_rho))
        return np.array([theta, sigma, phi, rho])

    def unconstrain_shape(self, shape):
        theta, sigma, phi, rho = shape
        lg_theta = np.log(theta) - np.log(1. - theta)
        lg_sigma = np.log(sigma)
        phi_unit = (phi+180) / 180.
        lg_phi   = np.log(phi_unit) - np.log(1. - phi_unit)
        lg_rho   = np.log(rho) - np.log(1. - rho)
        return np.array([lg_theta, lg_sigma, lg_phi, lg_rho])

    def make_star_unconstrained_logp(self):
        """ generate a log unnormalized posterior function for a star, and 
        return initial sample """
        # jointly resample fluxes and location
        def loglike(th):
            u, color = self.constrain_loc(th[:2]), th[2:]  #unpack params
            fluxes   = np.exp(star_flux_mog.to_fluxes(color))
            #ll       = self.log_likelihood(u=u, fluxes=fluxes)
            ll = self.background_image_loglike(u=u, fluxes=fluxes)
            ll_color = star_flux_mog.logpdf(color)
            return ll+ll_color
        gloglike = grad(loglike)
        # pack params (make sure we convert to color first
        def pack_params(params):
            lfluxes = np.log(params.fluxes)
            return np.concatenate([self.unconstrain_loc(params.u),
                                   star_flux_mog.to_colors(lfluxes)])
        def unpack_params(th):
            return self.constrain_loc(th[:2]), np.exp(star_flux_mog.to_fluxes(th[2:]))
        return loglike, gloglike, pack_params(self.params), unpack_params, pack_params

    def make_gal_unconstrained_logp(self):
        # gradient w.r.t fluxes
        def loglike(th):
            # unpack location, color and shape parameters
            u, color, shape = self.constrain_loc(th[:2]), th[2:7], \
                              self.constrain_shape(th[7:])
            fluxes          = np.exp(gal_flux_mog.to_fluxes(color))
            #ll              = self.log_likelihood(u=u, fluxes=fluxes, shape=shape)
            ll = self.background_image_loglike(u=u, fluxes=fluxes, shape=shape)
            ll_color        = gal_flux_mog.logpdf(color)
            return ll+ll_color
        gloglike = grad(loglike)

        #print "initial conditional likelihood: %2.4f"%loglike(th)
        def pack_params(params):
            params.theta = np.clip(params.theta, 1e-6, 1-1e-6)
            th  = np.concatenate([self.unconstrain_loc(params.u),
                                  gal_flux_mog.to_colors(np.log(params.fluxes)),
                                  self.unconstrain_shape(params.shape)])
            return th
        def unpack_params(th):
            u, fluxes, shape = ( self.constrain_loc(th[:2]),
                                 np.exp(gal_flux_mog.to_fluxes(th[2:7])),
                                 self.constrain_shape(th[7:]) )
            return u, fluxes, shape
        return loglike, gloglike, pack_params(self.params), unpack_params, pack_params

    def resample_star(self, eps=.001, num_steps=10, mass=1., P=None):
        # jointly resample fluxes and location
        loglike, gloglike, th, unpack_params, _ = self.make_star_unconstrained_logp()
        #print "initial conditional likelihood: %2.4f"%loglike(th)
        if P is None:
            P = np.sqrt(mass)*np.random.randn(th.shape[0])
        th, P, _, _ = hmc(th, loglike, gloglike, eps=eps, num_steps=num_steps,
                          mass=mass, p_curr = P)
        self.params.u, self.params.fluxes = unpack_params(th)
        return P

    def resample_galaxy(self, eps=.001, num_steps=10, mass=1., P=None):
        # gradient w.r.t fluxes
        loglike, gloglike, th, unpack_params, _ = self.make_gal_unconstrained_logp()
        if P is None:
            P = np.sqrt(mass)*np.random.randn(th.shape[0])
        th, P, _, _ = hmc(th, loglike, gloglike, eps=eps, num_steps=num_steps,
                          mass=mass, p_curr = P)
        # store new values
        self.params.u, self.params.fluxes, self.params.shape = unpack_params(th)
        return P

    def linear_propose_other_type(self):
        """ based on linear regression of fluxes and conditional distribution
        of galaxy shapes, propose parameters of the other type and report
        the log probability of generating that proposal

        Returns:
            - proposal params
            - log prob of proposal
            - log prob of implied reverse proposal
            - log determinant of the transformation |d(x',u')/d(x,u)|

        """
        params = SrcParams(u=self.params.u)
        if self.is_star():
            params.a = 1

            # fluxes
            residual       = mvn.rvs(cov=star_mag_proposal.res_covariance)
            ll_prop_fluxes = mvn.logpdf(residual, mean=None, cov=star_mag_proposal.res_covariance)
            gal_mag = star_mag_proposal.predict(self.params.mags.reshape((1,-1))) + residual
            params.fluxes = du.mags2nanomaggies(gal_mag).flatten()

            # compute reverse ll
            res   = gal_mag_proposal.predict(gal_mag) - self.params.mags
            llrev = mvn.logpdf(res, mean=None, cov=gal_mag_proposal.res_covariance)

            # shape
            sample_re = star_rad_proposal.rvs(size=1)[0]
            ll_shape  = star_rad_proposal.logpdf(sample_re)
            params.shape = np.array([np.random.rand(),
                                     np.exp(sample_re),
                                     np.random.rand() * np.pi,
                                     np.random.rand()])
            _, logdet = np.linalg.slogdet(star_mag_proposal.coef_)
            return params, ll_prop_fluxes + ll_shape, llrev, logdet

        elif self.is_galaxy():
            params.a = 0
            # fluxes
            residual = mvn.rvs(cov=gal_mag_proposal.res_covariance)
            llprob   = mvn.logpdf(residual, mean=None, cov=gal_mag_proposal.res_covariance)
            star_mag = gal_mag_proposal.predict(self.params.mags.reshape((1, -1))) + residual
            params.fluxes = du.mags2nanomaggies(star_mag).flatten()

            res   = star_mag_proposal.predict(star_mag) - self.params.mags
            llrev = mvn.logpdf(res, mean=None, cov=star_mag_proposal.res_covariance)
            ll_re = star_rad_proposal.logpdf(np.log(self.params.sigma))

            _, logdet = np.linalg.slogdet(gal_mag_proposal.coef_)
            return params, llprob, llrev + ll_re, logdet

    #def resample_type(self, proposal_fun=None):
    #    """ resample type of source - star vs. galaxy """
    #    # propose new parameter setting: 
    #    #  - return new params, probability of proposal, probability of reverse proposal
    #    #  - and log det of transformation from current to proposed parameters
    #    proposal_fun = self.propose_other_type_prior if proposal_fun is None else proposal_fun
    #    proposal, logpdf, logreverse, logdet = proposal_fun()

    #    fimgs = [self.model.field_list[0].img_dict[b] for b in self.model.bands]
    #    accept_logprob = self.calculate_acceptance_logprob(
    #            proposal, logpdf, logreverse, logdet, fimgs)
    #    print accept_logprob
    #    if np.log(np.random.rand()) < accept_logprob:
    #        self.params = proposal
    #        #TODO: keep stats of acceptance + num like evals

    #def propose_other_type_prior(self):
    #    """ prior-based proposal (simplest).  override this for more
    #    efficient proposals
    #    """
    #    if self.is_star():
    #        params, logprob = self.model.prior_sample('galaxy', u=self.params.u)
    #        #print "galaxy proposal:", params.fluxes, params.shape
    #    elif self.is_galaxy():
    #        params, logprob = self.model.prior_sample('star', u=self.params.u)
    #        #print "star proposal:", params.fluxes
    #    logreverse = self.model.logprior(self.params)
    #    return params, logprob, logreverse, 0.

    def calculate_acceptance_logprob(self, proposal, logprob_proposal,
            logprob_reverse, logdet, images):

        def image_like(src, img):
            # get biggest bounding box needed to consider for this image
            xlim, ylim     = self.bounding_boxes[img]
            background_img = self.background_image_dict[img]
            data_img       = img.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            mask_img       = img.invvar[ylim[0]:ylim[1], xlim[0]:xlim[1]]

            # model image for img, (xlim, ylim)
            model_img, _, _ = src.compute_model_patch(img, xlim=xlim, ylim=ylim)

            # compute current model loglike and proposed model loglike
            ll = poisson_loglike(data      = data_img,
                                 model_img = background_img+model_img,
                                 mask      = mask_img)
            return ll

        # compute current and proposal model likelihoods
        curr_like       = np.sum([image_like(self, img) for img in images])
        curr_logprior   = self.model.logprior(self.params)

        proposal_source = self.model._source_type(proposal, self.model, self.imgs)
        prop_like       = np.sum([image_like(proposal_source, img) for img in images])
        prop_logprior   = self.model.logprior(proposal_source.params)

        print """
            RJ acceptance breakdown:
                curr ({curr_type})
                    loglike: {curr_like}
                    lnprior: {curr_logprior}
                prop ({prop_type})
                    loglike: {prop_like}
                    lnprior: {prop_logprior}
                ln_curr - ln_prop    : {ll}
                logprob curr -> prop : {logprob_proposal}
                logprob prop -> curr : {logprob_reverse}
                logdet |prop(curr)|  : {logdet}
        """.format(curr_type = self.object_type,
                   curr_like = curr_like, curr_logprior = curr_logprior, 
                   prop_type = proposal_source.object_type,
                   prop_like = prop_like, prop_logprior = prop_logprior,
                   logprob_proposal = logprob_proposal, logprob_reverse = logprob_reverse,
                   logdet = logdet, ll = (curr_like+curr_logprior - prop_like-prop_logprior))

        # compute acceptance ratio
        accept_ll = (prop_like + prop_logprior) - (curr_like + curr_logprior) + \
                    (logprob_reverse - logprob_proposal) + \
                    logdet
        return accept_ll


#################################################################
# Subclass base model class, using the above source class       #
#################################################################

class CelesteGMMPrior(CelesteBase):
    _source_type = SourceGMMPrior

    def __init__(self, images,
                       star_flux_prior   = star_flux_mog,
                       galaxy_flux_prior = gal_flux_mog,
                       galaxy_re_prior   = gal_re_mog,
                       galaxy_ab_prior   = gal_ab_mog):
        self.star_flux_prior    = star_flux_prior
        self.galaxy_flux_prior  = galaxy_flux_prior
        self.galaxy_re_prior    = galaxy_re_prior
        self.galaxy_ab_prior    = galaxy_ab_prior
        super(CelesteGMMPrior, self).__init__(images=images)

    def logprior(self, params):
        if params.is_star():
            color = self.star_flux_prior.to_colors(params.fluxes)
            return self.star_flux_prior.logpdf(color)
        elif params.is_galaxy():
            color = self.galaxy_flux_prior.to_colors(params.fluxes)
            return self.galaxy_flux_prior.logpdf(color) + 0.
                    # todo include constraints for shape parameters

    def prior_sample(self, src_type, u=None):
        params = SrcParams(u=u)
        if src_type == 'star':
            # TODO SET a with atoken
            params.a = 0
            color   = self.star_flux_prior.rvs(size=1)[0]
            logprob = self.star_flux_prior.logpdf(color)
            params.fluxes = np.exp(self.star_flux_prior.to_fluxes(color))
            return params, logprob

        elif src_type == 'galaxy':
            params.a = 1
            color   = self.galaxy_flux_prior.rvs(size=1)[0]
            logprob = self.galaxy_flux_prior.logpdf(color)
            params.fluxes = np.exp(self.galaxy_flux_prior.to_fluxes(color))

            sample_ab = self.galaxy_ab_prior.rvs(size=1)[0,0]
            sample_ab = np.exp(sample_ab) / (1.+np.exp(sample_ab))

            params.shape  = np.array([np.random.random(),
                                      np.exp(self.galaxy_re_prior.rvs(size=1)[0,0]),
                                      np.random.random() * np.pi,
                                      sample_ab])

            logprob_re    = self.galaxy_re_prior.logpdf(params.sigma)
            logprob_ab    = self.galaxy_ab_prior.logpdf(params.rho)
            logprob_shape = -np.log(np.pi) + logprob_re + logprob_ab
            return params, logprob


