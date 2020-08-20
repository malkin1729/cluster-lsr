import numpy as np
import scipy.io

nlcd_cl = [ 0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255 ]
c2i = {cl:i for i,cl in enumerate(nlcd_cl)}

ncolors = np.zeros((len(nlcd_cl),3))
ncolors[1:-1] = scipy.io.loadmat('data/nlcd_legend.mat')['nlcd_cmap'][:,1:] / 255.
def vis_nlcd(r, sparse=False, renorm=True):
    if sparse: r = np.array([(r==nlcd_cl[i]) for i in range(22)])
    z = np.zeros((3,) + r.shape[1:])
    if renorm: s = r / r.sum(0)
    else: s = r
    for c in range(22):
        for ch in range(3):
            z[ch] += ncolors[c,ch] * s[c]
    return z
