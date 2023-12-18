""" NMF Analysis """

import numpy as np


from ihop.iops import pca as ihop_pca

from cnmf.oceanography import iops
from cnmf import nmf_imaging
from cnmf import io as cnmf_io
from cnmf import zhu_nmf as nmf

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
    cnmf_io.save_nmf(outfile, M, coeff, spec_nw[...,0],
                     mask[...,0], err[...,0], wave, Rs)

    print(f'Wrote: {outfile}')

def l23_nmf_on_tara(sig:float=0.0005,
                    cut:int=None):

    # Load L23 fit
    nmf_fit, N_NMF, iop = 'L23', 4, 'a'
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    # Calculate Tara
    wv_grid, final_tara, l23_a = iops.tara_matched_to_l23(
        low_cut=410.)
    i0 = np.argmin(np.abs(wv_grid[0]-wave))
    assert np.isclose(wv_grid[0], wave[i0])
    i1 = np.argmin(np.abs(wv_grid[-1]-wave))

    # Cut?
    if cut is not None:
        final_tara = final_tara[:cut]
    V = np.ones_like(final_tara) / sig**2
    M_tara = M[:,i0:i1+1]

    # Build it up one component at a time
    H_tmp = None
    for nn in range(M_tara.shape[0]):
        print("Working on component: ", nn+1)
        W_ini = M_tara[0:nn+1,:].T
        H_rand = np.random.rand(1, final_tara.shape[0])
        if H_tmp is not None:
            H_ini = np.vstack((H_tmp, H_rand))
        else:
            H_ini = H_rand
    
        tara_NMF = nmf.NMF(final_tara.T,
                       V=V.T, W=W_ini, H=H_ini,
                       n_components=nn+1)
        # Do it
        tara_NMF.SolveNMF(H_only=True, verbose=True)
        # Save H
        H_tmp = tara_NMF.H.copy()
        embed(header='iops 84')

    # Save
    outfile = cnmf_io.nmf_filename('Tara_L23', N_NMF=N_NMF, iop=iop)
    cnmf_io.save_nmf(outfile, M_tara, tara_NMF.H, final_tara,
                     None, V, wv_grid, None)

if __name__ == '__main__':


    '''
    # NMF
    for n in range(1,10):
        loisel23_components('a', N_NMF=n+1)
        loisel23_components('bb', N_NMF=n+1)


    #PCA
    ihop_pca.generate_l23_pca(clobber=False, Ncomp=20, X=4, Y=0)
    ihop_pca.generate_l23_pca(clobber=False, Ncomp=4, X=4, Y=0)
    '''

    # L23 NMF on Tara
    l23_nmf_on_tara(cut=20000)