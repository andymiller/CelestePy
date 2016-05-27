import autograd.numpy as np

#####################################################
# simple converter between mags and nanomaggies     #
#####################################################

def mags2nanomaggies(mags):
    return np.power(10., (mags - 22.5)/-2.5)

def nanomaggies2mags(nanos):
    return (-2.5)*np.log10(nanos) + 22.5


###########################################################
# convert between mags and colors  (for sdss ugriz bansd) #
###########################################################
colors = ['ug', 'gr', 'ri', 'iz']

def mags_to_colors(mags, ridx = 2):
    rmag = mags[ridx]
    colors = np.diff(mags[::-1])[::-1]
    return rmag, colors


def colors_to_mags(rmag, colors):
    ug, gr, ri, iz = colors
    g = gr + rmag
    u = ug + g
    i = rmag - ri
    z = i - iz
    return np.array([u, g, rmag, i, z])



