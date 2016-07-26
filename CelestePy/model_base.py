"""
CelestePy catalog of the universe.

Defines the template for the base class of a `Celeste` model object and some
base-level functions.  Celeste Models are a way to organize a list of many
sources and manipulate global parameters.

Andrew Miller <acm@seas.harvard.edu>
"""
import autograd.numpy as np
import CelestePy.sample.gibbs_steps as cel_mcmc
import CelestePy.util.data as du
import pyprind

class CelesteBase(object):
    """ Main model class - interface to user.  Holds a list of
    Source objects, each of which contains local markov chain state, 
    including source parameters, source-sample images"""
    def __init__(self, images):
        #TODO incorporate priors over fluxes
        # model keeps track of a field list
        self.images     = images
        self.field_list = []
        self.bands = ['u', 'g', 'r', 'i', 'z']

    def initialize_sources(self, init_srcs=None, init_src_params=None, photoobj_df=None):
        """ initialize sources after adding fields """
        if init_srcs is not None:
            self.srcs = init_srcs
        elif init_src_params is not None:
            self.srcs = [self._source_type(s, self) for s in init_src_params]
        elif photoobj_df is not None:
            self.srcs = [self._source_type(du.photoobj_to_celestepy_src(p), self)
                         for (i, p) in photoobj_df.iterrows()]
        else:
            raise NotImplementedError

    def intialize_from_fields(self):
        """initialize sources from bright spots - make everything a point source?"""
        raise NotImplementedError

    def init_celeste_source_from_df(self, celestedf_row, is_star=None):
        """
        Initialize a celeste source object from a celeste_df row
        ----------
        Args:
            celestedf_row : row from a pandas dataframe of celeste
                df parameters
            is_star       : if True/False, force the initialized object to be a
                star/gal.  If None, will be whatever the row says
        """
        if is_star in [True, False]:
            celestedf_row.is_star = is_star
        params = du.celestedf_row_to_params(celestedf_row)
        src    = self._source_type(params, model=self, imgs=self.images)
        # add on some more info for tracking
        src.objid  = celestedf_row.objid
        src.run    = celestedf_row.run
        src.camcol = celestedf_row.camcol
        src.field  = celestedf_row.field
        return src

    def add_field(self, img_dict):
        """ add a field (run/camcol/field) image information to the model
                img_dict  = fits image keyed by 'ugriz' band
                init_srcs = sources (tractor or celeste) initialized in this field
        """
        for k in img_dict.keys():
            assert k in self.bands, "Celeste model doesn't support band %s"%k
        self.field_list.append(Field(img_dict))

    @property
    def source_types(self):
        return np.array([s.object_type for s in self.srcs])

    def get_brightest(self, object_type='star', num_srcs=1, band='r', return_idx=False):
        """return brightest sources (by source type, band)"""
        fluxes      = np.array([s.params.flux_dict[band] for s in self.srcs])
        type_idx    = np.where(self.source_types == object_type)[0]
        type_fluxes = fluxes[type_idx]
        type_idx    = type_idx[np.argsort(type_fluxes)[::-1]][:num_srcs]
        blist       = [self.srcs[i] for i in type_idx]
        if return_idx:
            return blist, type_idx
        else:
            return blist

    ####################
    # Resample methods #
    ####################

    def resample_model(self):
        """ resample each field """
        for field in pyprind.prog_bar(self.field_list):
            field.resample_photons(self.srcs)
        self.resample_sources()

    def resample_sources(self):
        for src in pyprind.prog_bar(self.srcs):
            src.resample()

    #####################
    # Plotting Methods  #
    #####################
    def render_model_image(self, fimg, xlim=None, ylim=None, exclude=None):
        # create model image, and add each patch in - init with sky noise
        mod_img     = np.ones(fimg.nelec.shape) * fimg.epsilon
        source_list = [s for s in self.srcs if s is not exclude]

        if not len(source_list) == 0:
            # add each source's model patch
            for s in pyprind.prog_bar(source_list):
                patch, ylim, xlim = s.compute_model_patch(fits_image=fimg, xlim=xlim, ylim=ylim)
                mod_img[ylim[0]:ylim[1], xlim[0]:xlim[1]] += patch

        if xlim is not None and ylim is not None:
            mod_img = mod_img[ylim[0]:ylim[1], xlim[0]:xlim[1]]

        return mod_img

    #def img_log_likelihood(self, fimg, mod_img=None):
    ##    if mod_img is None:
    #        mod_img = self.render_model_image(fimg)
    #    ll = np.sum(np.log(mod_img) * fimg.nelec) - np.sum(mod_img)
    #    return ll

class Field(object):
    """ holds image data associated with a single field """
    def __init__(self, img_dict):
        self.img_dict = img_dict

        # set each image noise level to the median
        for k, img in self.img_dict.iteritems():
            img.epsilon = np.median(img.nelec)

        # set the (gamma) prior over noise level
        self.a_0 = 5      # convolution parameter - higher tends to avoid 0
        self.b_0 = .005   # inverse scale parameter

    def resample_photons(self, srcs, verbose=False):
        """resample photons - store source-specific images"""
        # first, clear out old sample images
        for src in srcs:
            src.clear_sample_images()

        # generate per-source sample image patch for each fits image in
        # this field.  keep track of photons due to noise
        noise_sums = {}
        for band, img in self.img_dict.iteritems():
            if verbose:
                print " ... resampling band %s " % band
            samp_imgs, noise_sum = \
                cel_mcmc.sample_source_photons_single_image_cython(
                    img, [s.params for s in srcs]
                )

            # tell each source to keep track of it's source-specific sampled
            # images (and the image it was stripped out of)
            for src, samp_img in zip(srcs, samp_imgs):
                if samp_img is not None:

                    # cache pixel grid for each sample image
                    y_grid = np.arange(samp_img.y0, samp_img.y1, dtype=np.float)
                    x_grid = np.arange(samp_img.x0, samp_img.x1, dtype=np.float)
                    xx, yy = np.meshgrid(x_grid, y_grid, indexing='xy')
                    pixel_grid = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C')))
                    src.sample_image_list.append((samp_img, img, pixel_grid))

            # keep track of noise sums
            noise_sums[band] = noise_sum

        # resample noise parameter in each fits image
        for band, img in self.img_dict.iteritems():
            a_n         = self.a_0 + noise_sums[band]
            b_n         = self.b_0 + img.nelec.size
            #eps_tmp     = img.epsilon
            img.epsilon = np.random.gamma(a_n, 1./b_n)


def get_active_sources(source, source_list, image):
    """Given an initial source, "source" with a bounding box, find all
    sources in source_list where their bounding box intersects with "source"'s
    bounding box.

    Collect all sources that contribute to this source's background model image
    """
    def intersect(sa, sb, image):
        xlima, ylima = sa.bounding_boxes[image]
        xlimb, ylimb = sb.bounding_boxes[image]
        widtha, heighta = xlima[1] - xlima[0], ylima[1] - ylima[0]
        widthb, heightb = xlimb[1] - xlimb[0], ylimb[1] - ylimb[0]
        return (np.abs(xlima[0] - xlimb[0])*2 < (widtha + widthb)) and \
               (np.abs(ylima[0] - ylimb[0])*2 < (heighta + heightb))
    return [s for s in source_list if intersect(s, source, image) and s is not source]


def generate_background_patch(source, source_list, image):
    active_sources = get_active_sources(source, source_list, image)
    xlim, ylim     = source.bounding_boxes[image]
    if len(active_sources) < 1:
        return image.epsilon * np.ones((ylim[1]-ylim[0], xlim[1]-xlim[0]))
    background     = np.sum([s.compute_model_patch(image, xlim=xlim, ylim=ylim)[0]
                             for s in active_sources], axis=0) + image.epsilon
    return background


#####################################
# Instantiate Generic Celeste class #
#####################################
from source_base import Source
class Celeste(CelesteBase):
    _source_type = Source

