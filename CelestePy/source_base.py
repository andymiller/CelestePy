import autograd.numpy as np
import CelestePy.celeste_funs as cel_funs
from CelestePy.util.transform import nanomaggies2mags
from CelestePy.util.dists.poisson import poisson_loglike

SDSS_BANDS = ['u', 'g', 'r', 'i', 'z']

class Source(object):
    """
    Base class for sources.
    Contains methods for rendering images of sources, computing model
    images, and keeping track of MCMC source samples
    Arguments
    ----------
    params: SrcParams object
        Struct that holds the current values of this source's params
    model: Celeste Model object
    Notes
    -----
    """
    def __init__(self, params, model, imgs, pixel_radius=30):
        self.params = params
        self.model  = model
	self.imgs   = imgs
        self.sample_image_list = []

        # bounding box dict for each source - heuristically set for now
        self.bounding_boxes = make_bbox_dict(self.params, imgs, pixel_radius=30)

        # create a bounding box of ~ 20x20 pixels to bound location
        # we're assuming we're within 20 pixels on the first guess...
        self.u_lower = self.params.u - .0025
        self.u_upper = self.params.u + .0025
        self.du      = self.u_upper - self.u_lower

        # kept samples for each source
        self.loc_samps      = []
        self.flux_samps     = []
        self.shape_samps    = []
        self.type_samps     = []
        self.ll_samps       = []

    def clear_sample_images(self):
        self.sample_image_list = []
        #TODO maybe force a garbage collect?

    def __str__(self):
        label = "%s at Ra, Dec = (%2.5f, %2.5f) " % \
            (self.object_type, self.params.u[0], self.params.u[1])
        mags  = " ".join([ "%s = %2.4f"%(b, nanomaggies2mags(f))
                           for b,f in zip(['u', 'g', 'r', 'i', 'z'],
                                           self.params.fluxes) ])
        if self.is_star():
            return label + " with Mags " + mags
        else:
            shape = "re=%2.3f, ab=%2.3f, phi=%2.3f" % \
                (self.params.sigma, self.params.rho, self.params.phi)
            return label + " with Mags " + mags + " and Galaxy Shape: " + shape

    @property
    def object_type(self):
        if self.is_star():
            return "star"
        elif self.is_galaxy():
            return "galaxy"
        else:
            return "none"

    def is_star(self):
        return self.params.a == 0

    def is_galaxy(self):
        return self.params.a == 1

    @property
    def id(self):
        return "in_%s_to_%s"%(np.str(self.u_lower), np.str(self.u_upper))

    ############################################
    # model image rendering functions          #
    ############################################

    def compute_scatter_on_pixels(self, fits_image, u=None, shape=None,
                                        xlim=None, ylim=None,
                                        pixel_grid=None, force_type=None):
        """ compute how photons will be scattered spatially on fits_image, 
        subselecting only the pixels with  > epsilon probability of seeing a 
        photon.
        For a star, this is just the PSF image.  For a Galaxy, this
        is the convolution of the PSF model with the PSF

        kwargs: 
          u          : source (ra, dec) location
          shape      : galaxy shape parameters
          xlim       : pixel limits to compute scatter (must be within fits_image bounds)
          ylim       : ''
          pixel_grid : list of points to evaluate mog (cached for speed)
        """
        #TODO make sure this returns an image with EPSILON error - 
        u       = self.params.u if u is None else u
        render_star = self.is_star() if force_type is None else (force_type=='star')
        render_gal  = self.is_galaxy() if force_type is None else (force_type=='galaxy')
        if render_star:
            patch, ylim, xlim = \
                cel_funs.gen_point_source_psf_image(u, fits_image,
                                                   xlim=xlim, ylim=ylim,
                                                   pixel_grid=pixel_grid)
            return patch, ylim, xlim
        elif render_gal:
            if shape is None:
                shape = self.params.shape
            #print shape
            patch, ylim, xlim = \
                cel_funs.gen_galaxy_psf_image(shape, u, fits_image,
                                              xlim=xlim, ylim=ylim,
                                              check_overlap=True,
                                              unconstrained=False,
                                              return_patch=True)
            return patch, ylim, xlim
        else:
            raise NotImplementedError, "only stars and galaxies have photon scattering images"

    def compute_model_patch(self, fits_image, u=None, fluxes=None, shape=None,
                            xlim=None, ylim=None):
        """creates a model image patch for the given fits image"""
        # compute unit flux model
        patch, ylim, xlim = \
            self.compute_scatter_on_pixels(fits_image, u=u, shape=shape,
                                           xlim=xlim, ylim=ylim)
        # scale by flux parameters
        if fluxes is None:
            fluxes = self.params.fluxes
        bflux     = fluxes[SDSS_BANDS.index(fits_image.band)]
        band_flux = (bflux / fits_image.calib) * fits_image.kappa
        return band_flux * patch, ylim, xlim

    def flux_in_image(self, fits_image, fluxes=None):
        """convert flux in nanomaggies to flux in pixel counts for a 
        particular image """
        if fluxes is not None:
            band_i = ['u', 'g', 'r', 'i', 'z'].index(fits_image.band)
            f      = fluxes[band_i]
        else:
            f = self.params.flux_dict[fits_image.band]
        band_flux = (f / fits_image.calib) * fits_image.kappa
        return band_flux

    ###############################################################
    # source store their own location, fluxes, and shape samples  #
    ###############################################################
    @staticmethod
    def get_bounding_box(params, img):
        if params.is_star():
            bound = img.R
        elif params.is_galaxy():
            bound = cel_funs.gen_galaxy_psf_image_bound(params, img)
        else:
            raise "source type unknown"
        px, py = img.equa2pixel(params.u)
        xlim = (np.max([0,                  np.floor(px - bound)]),
                np.min([img.nelec.shape[1], np.ceil(px + bound)]))
        ylim = (np.max([0,                  np.floor(py - bound)]),
                np.min([img.nelec.shape[0], np.ceil(py + bound)]))
        return xlim, ylim

    ########################################
    # store and assemble sample functions  #
    ########################################

    @property
    def location_samples(self):
        return np.array(self.loc_samps)

    @property
    def flux_samples(self):
        return np.array(self.flux_samps)

    @property
    def shape_samples(self):
        return np.array(self.shape_samps)

    @property
    def type_samples(self):
        return np.array(self.type_samps)

    @property
    def loglike_samples(self):
        return np.array(self.ll_samps)

    def store_sample(self):
    	self.type_samps.append(self.params.a.copy())
        self.loc_samps.append(self.params.u.copy())
        self.flux_samps.append(self.params.fluxes.copy())
        self.shape_samps.append(self.params.shape.copy())

    def store_loglike(self):
        self.ll_samps.append(self.log_likelihood())

    ##########################################################################
    # likelihood functions (used by resampling methods, can be overridden)   #
    ##########################################################################

    def image_loglike(self, u, fluxes, shape=None, bounding_boxes=None,
                      background_image_dict=None, data_img_dict=None,
                      mask_img_dict=None):
        """
        Compute the log likelihood of current model parameters
        ----------------
        Arguments:
            u     : np.array, location in equatorial coordinates
            fluxes: np.array, [ugriz] fluxes
            shape : np.array([self.theta, self.sigma, self.phi, self.rho])
            bounding_boxes: dict of bounding boxes, keyed by FitsImage
            background_image_dict: dict of pre-computed background images,
                keyed by FitsImage
            data_img_dict: dict of images of observations, keyed by FitsImage
            mask_img_dict: dict of mask images of observations, keyed by
                FitsImage
        TODO: incorporate data + mask image dictionary defaults
        """
        def image_like(img):
            # get biggest bounding box needed to consider for this image
            xlim, ylim     = bounding_boxes[img]
            background_img = background_image_dict[img]
            data_img       = img.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            mask_img       = img.invvar[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            # model image for img, (xlim, ylim)
            model_img, _, _ = \
                self.compute_model_patch(img, xlim=xlim, ylim=ylim, u=u,
                                         fluxes=fluxes, shape=shape)
            # compute current model loglike and proposed model loglike
            ll = poisson_loglike(data      = data_img,
                                 model_img = background_img+model_img,
                                 mask      = mask_img)
            return ll
        imgs = self.bounding_boxes.keys()
        return np.sum([image_like(img) for img in imgs])


    def background_image_loglike(self, u, fluxes, shape=None):
        return self.image_loglike(u, fluxes, shape,
                             bounding_boxes=self.bounding_boxes,
                             background_image_dict=self.background_image_dict)

    def log_likelihood(self):
        return self.image_loglike(self.params.u,
                                 self.params.fluxes,
                                 self.params.shape,
                             bounding_boxes=self.bounding_boxes,
                             background_image_dict=self.background_image_dict)

    #############################################################
    # resample params - must be implemented in subclass         #
    #############################################################

    def resample(self):
        raise NotImplementedError


    ###################
    # source plotting #
    ###################

    def plot(self, fits_image, ax, data_ax=None, diff_ax=None, unit_flux=False):
        import matplotlib.pyplot as plt; import seaborn as sns;
        import CelestePy.util.plots as plot_util
        if unit_flux:
            patch, ylim, xlim = self.compute_scatter_on_pixels(fits_image)
        else:
            patch, ylim, xlim = self.compute_model_patch(fits_image)
        cim = ax.imshow(patch, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
        plot_util.add_colorbar_to_axis(ax, cim)
        ax.set_title("model")

        if data_ax is not None:
            dpatch = fits_image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]].copy()
            print "Data patch median: ", np.median(dpatch)
            dpatch -= np.median(dpatch)
            dpatch[dpatch<0] = 0.
            dim = data_ax.imshow(dpatch, extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
            plot_util.add_colorbar_to_axis(data_ax, dim)
            data_ax.set_title("data")

        if diff_ax is not None:
            dpatch = fits_image.nelec[ylim[0]:ylim[1], xlim[0]:xlim[1]].copy()
            dpatch -= np.median(dpatch)
            dpatch[dpatch<0] = 0.
            dim = diff_ax.imshow((dpatch - patch), extent=(xlim[0], xlim[1], ylim[0], ylim[1]))
            plot_util.add_colorbar_to_axis(diff_ax, dim)
            msqe  = np.mean((dpatch - patch)**2)
            smsqe = np.mean((dpatch - patch)**2 / patch)
            diff_ax.set_title("diff, mse = %2.3f"%msqe)


#####################################################################
# some source utility functions
#####################################################################
def make_bbox_dict(params, images, pixel_radius=None):
    """ for a set of model parameters and a list of images create a dictionary
    from {img: bounding box}, where each entry describes the area affected
    by this source's model image

    Args:
        - params       : SrceParams or SrcMixParams (anything with a u)
        - images       : list of FitsImages
        - pixel_radius : (optional)

    Returns:
        - dict: {img : bbox}
    """
    if pixel_radius is None:
        raise NotImplementedError
    def image_bbox(params, img):
        img_ymax, img_xmax = img.nelec.shape
        px, py = img.equa2pixel(params.u)
        xlim = (np.max([0,        int(np.floor(px - pixel_radius))]),
                np.min([img_xmax, int(np.ceil(px + pixel_radius))]))
        ylim = (np.max([0,        int(np.floor(py - pixel_radius))]),
                np.min([img_ymax, int(np.ceil(py + pixel_radius))]))
        return xlim, ylim
    return {img: image_bbox(params, img) for img in images}


