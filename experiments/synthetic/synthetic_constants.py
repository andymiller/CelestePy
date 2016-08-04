from CelestePy.util.dists import mog
import autograd.numpy as np
import cPickle as pickle
import CelestePy.util.data as data_util
from CelestePy.util.transform import fluxes_to_colors
from CelestePy.point_source import bands

try:

    img_constants = pickle.load(open('synth_constants.pkl', 'rb'))

except:

    print "need to run this as main first!"


# TODO have a sample from prior option for these
def synthetic_star_params():
    """ generates some star parameters for a UGRIZ image """
    # true params - scale eps_dict by noise_scale arg
    u = img_constants['r']['phi'] + 1./3600.  # near center, off by a pixel
    flux_dict = {'g': 9.6173432087297002,
                 'i': 33.070941854638555,
                 'r': 24.437380835296388,
                 'u': 1.2582444245272928,
                 'z': 40.854689375715807}
    true_colors = fluxes_to_colors(np.array([flux_dict[b] for b in bands]))
    true_params = np.concatenate([true_colors, u])
    return true_params, true_colors



if __name__=="__main__":

    data_dict          = pickle.load(open('s82_dict.pkl', 'rb'))
    run, camcol, field = data_dict['run'], data_dict['camcol'], data_dict['field']
    primary_field_df   = data_dict['primary_field_df']
    coadd_field_df     = data_dict['coadd_field_df']

    # load in fits images
    imgfits = data_util.make_fits_images(run, camcol, field)
    for k, img in imgfits.iteritems():
        img.epsilon = np.median(img.nelec)

    # extract constants needed from each img
    def extract_constants(fimg):
        return  {'Ups'             : fimg.Ups_n,
                 'Ups_inv'         : fimg.Ups_n_inv,
                 'phi'             : fimg.phi_n,
                 'psf'             : fimg.psf,
                 'calib'           : fimg.calib,
                 'kappa'           : fimg.kappa,
                 'photons_per_nmgy': fimg.kappa / fimg.calib,
                 'band'            : fimg.band}
    img_constants = { k: extract_constants(fimg) for k,fimg in imgfits.iteritems() }
    with open('synth_constants.pkl', 'wb') as f:
        pickle.dump(img_constants, f)

