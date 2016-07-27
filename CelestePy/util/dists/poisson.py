import autograd.numpy as np

def poisson_loglike(data, model_img, mask):
    assert model_img.shape == mask.shape
    assert data.shape == model_img.shape
    good_pix = (model_img > 0.) & (mask != 0)
    ll_img   = np.sum(np.log(model_img[good_pix]) * data[good_pix]) - \
               np.sum(model_img[good_pix])
    return ll_img

def poisson_logpmf(x, lam, mask=None):
    assert x.shape == lam.shape
    good = (lam > 0.) if mask is None else (lam > 0.) & (mask != 0)
    return x[good] * np.log(lam[good]) - lam[good]
