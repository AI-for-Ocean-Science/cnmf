""" NMF Analysis """

import os
from importlib import resources

import numpy as np

from oceancolor.iop import cross

from ihop.iops import pca as ihop_pca

from cnmf.oceanography import iops
from cnmf import nmf_imaging
from cnmf import io as cnmf_io

from IPython import embed

def loisel23_components(iop:str, N_NMF:int=10):

    outfile = cnmf_io.nmf_filename('L23', N_NMF=N_NMF, iop=iop)
    outroot = outfile.replace('.npz','')

    # Load
    spec_nw, mask, err, wave, Rs = iops.prep_loisel23(iop)

    # Do it
    comps = nmf_imaging.NMFcomponents(
        ref=spec_nw, mask=mask, ref_err=err,
        n_components=N_NMF,
        path_save=outroot, oneByOne=True)

    # Load
    M = np.load(outroot+'_comp.npy').T
    coeff = np.load(outroot+'_coef.npy').T

    # Save
    np.savez(outfile, M=M, coeff=coeff,
             spec=spec_nw[...,0],
             mask=mask[...,0],
             err=err[...,0],
             wave=wave,
             Rs=Rs)

    print(f'Wrote: {outfile}')


if __name__ == '__main__':

    '''
    # NMF
    for n in range(1,10):
        loisel23_components('a', N_NMF=n+1)
        loisel23_components('bb', N_NMF=n+1)
    '''

    #PCA
    ihop_pca.generate_l23_pca(clobber=False, Ncomp=20, X=4, Y=0)
    ihop_pca.generate_l23_pca(clobber=False, Ncomp=4, X=4, Y=0)