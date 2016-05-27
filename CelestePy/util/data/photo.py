import autograd.numpy as np
from CelestePy.util.transform import mags2nanomaggies
from CelestePy.source_params import SrcParams

BANDS = ['u', 'g', 'r', 'i', 'z']

def photoobj_to_celestepy_src(photoobj_row):
    """Conversion between tractor source object and our source object...."""
    u = photoobj_row[['ra', 'dec']].values

    # brightnesses are stored in mags (gotta convert to nanomaggies)
    mags   = photoobj_row[['psfMag_%s'%b for b in ['u', 'g', 'r', 'i', 'z']]].values
    fluxes = [mags2nanomaggies(m) for m in mags]

    # photoobj type 3 are gals, type 6 are stars
    if photoobj_row.type == 6:
        return SrcParams(u, a=0, fluxes=fluxes)
    else:

        # compute frac dev/exp 
        prob_dev = np.mean(photoobj_row[['fracDeV_%s'%b for b in BANDS]])

        # galaxy A/B estimate, angle, and radius
        devAB      = np.mean(photoobj_row[['deVAB_%s'%b for b in BANDS]])
        devRad     = np.mean(photoobj_row[['deVRad_%s'%b for b in BANDS]])
        devPhi     = np.mean(photoobj_row[['deVPhi_%s'%b for b in BANDS]])
        expAB      = np.mean(photoobj_row[['expAB_%s'%b for b in BANDS]])
        expRad     = np.mean(photoobj_row[['expRad_%s'%b for b in BANDS]])
        expPhi     = np.mean(photoobj_row[['expPhi_%s'%b for b in BANDS]])

        #estimate - mix over frac dev/exp
        AB  = prob_dev * devAB + (1. - prob_dev) * expAB
        Rad = prob_dev * devRad + (1. - prob_dev) * expRad
        Phi = prob_dev * devPhi + (1. - prob_dev) * expPhi

        ## galaxy flux esimates
        devFlux = np.array([mags2nanomaggies(m) for m in
                                photoobj_row[['deVMag_%s'%b for b in BANDS]]])
        expFlux = np.array([mags2nanomaggies(m) for m in
                                photoobj_row[['expMag_%s'%b for b in BANDS]]])
        fluxes = prob_dev * devFlux + (1. - prob_dev) * expFlux

        #theta : exponential mixture weight. (1 - theta = devac mixture weight)
        #sigma : radius of galaxy object (in arcsc > 0)
        #rho   : axis ratio, dimensionless, in [0,1]
        #phi   : radians, "E of N" 0=direction of increasing Dec, 90=direction of increasting RAab
        return SrcParams(u,
                         a      = 1,
                         v      = u,
                         theta  = 1.0-prob_dev,
                         phi    = -1.*Phi, #(Phi * np.pi / 180.), # + np.pi / 2) % np.pi,
                         sigma  = Rad,
                         rho    = AB,
                         fluxes = fluxes)


