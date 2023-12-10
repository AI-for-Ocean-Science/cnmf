""" Figuers for the NMF paper"""

import os
from importlib import resources

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

import seaborn as sns
import pandas

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
mpl.rcParams['font.family'] = 'stixgeneral'

import corner

from oceancolor.utils import plotting 
from oceancolor.iop import cdom

from ihop.iops import pca as ihop_pca

from cnmf import io as cnmf_io
from cnmf import stats as cnmf_stats


from IPython import embed



# #############################################
def fig_l23_pca_nmf_var(
    outfile='fig_l23_pca_nmf_var.png',
    show_spec:bool=False, show_RMSE:bool=False,
    nmf_fit:str='L23'):

    # Load up
    L23_pca_N20 = ihop_pca.load('pca_L23_X4Y0_a_N20.npz')
    #L23_Tara_pca = ihop_pca.load(f'pca_L23_X4Y0_Tara_a_N{N}.npz')
    #wave = L23_pca_N20['wavelength']


    # Variance in NMF
    evar_list, index_list = [], []
    for i in range(2, 11):
        # Load
        d = cnmf_io.load_nmf(nmf_fit, i, 'a')
        # eval
        evar_i = cnmf_stats.evar_computation(
            d['spec'], d['coeff'], d['M'])
        evar_list.append(evar_i)
        index_list.append(i)

    # Figure
    clrs = ['b', 'g']
    figsize=(6,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # #####################################################
    # PCA
    ax= plt.subplot(gs[0])

    ax.plot(
        np.arange(L23_pca_N20['explained_variance'].size-1)+2,
        1-np.cumsum(L23_pca_N20['explained_variance'])[1:], 'o-',
        color=clrs[0])
    ax.plot(index_list, 1-np.array(evar_list), 'o-', color=clrs[1])

    ax.set_xlabel('Number of Components') 
    ax.set_ylabel('Cumulative Unexplained Variance')
    # Horizontal line at 1
    ax.axhline(1., color='k', ls=':')



    plotting.set_fontsize(ax, 17)
    ax.set_xlim(1., 10)
    ax.set_ylim(1e-5, 0.01)
    ax.set_yscale('log')
    for ss in range(2):
        lbl = 'PCA' if ss == 0 else 'NMF'
        ax.text(0.95, 0.90-ss*0.1, lbl, color=clrs[ss],
        transform=ax.transAxes,
            fontsize=22, ha='right')

    ax.text(0.05, 0.90, 'Loisel+2023', color='k',
        transform=ax.transAxes,
        fontsize=22, ha='left')

    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_nmf_pca_basis(outfile:str='fig_nmf_pca_basis.png',
                 nmf_fit:str='L23', Ncomp:int=4,
                 norm:bool=True):

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(1,2)

    # a, bb
    for ss, itype in zip([0,1], ['PCA', 'NMF']):

        # load
        if ss == 0:
            ab, Rs, d, d_bb = ihop_pca.load_loisel_2023_pca(N_PCA=Ncomp)
            wave = d['wavelength']
        elif ss == 1:
            d = cnmf_io.load_nmf(nmf_fit, Ncomp, 'a')
            wave = d['wave']
        M = d['M']
        #embed(header='fig_nmf_pca_basis 376')

        ax = plt.subplot(gs[ss])

        # Plot
        for ii in range(Ncomp):
            # Normalize
            if norm:
                iwv = np.argmin(np.abs(wave-440.))
                nrm = M[ii][iwv]
            else:
                nrm = 1.
            ax.step(wave, M[ii]/nrm, label=f'{itype}:'+r'  $\xi_'+f'{ii+1}'+'$')

        ax.set_xlabel('Wavelength (nm)')

        lbl = 'PCA' if ss == 0 else 'NMF'
        ax.set_ylabel(lbl+' Basis Functions')

        ax.legend(fontsize=15)


        if ss == 0:
            xlbl, ha, flbl = 0.95, 'right', '(a)'
        else:
            xlbl, ha, flbl = 0.05, 'left', '(b)'

        ax.text(xlbl, 0.05, flbl, color='k',
            transform=ax.transAxes,
              fontsize=18, ha=ha)

        plotting.set_fontsize(ax, 18)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_rmse(outfile:str='fig_nmf_rmse.png',
                 nmf_fit:str='L23'):

    # RMSE
    rmss = []
    for n in range(1,10):
        # load
        d = cnmf_io.load_nmf(nmf_fit, N_NMF=n+1, iop='a')
        N_NMF = d['M'].shape[0]
        recon = np.dot(d['coeff'],
                       d['M'])
        #
        dev = recon - d['spec']
        rms = np.std(dev, axis=1)
        # Average
        avg_rms = np.mean(rms)
        rmss.append(avg_rms)

    # Plot

    fig = plt.figure(figsize=(6,6))
    plt.clf()
    ax = plt.gca()

    ax.plot(2+np.arange(N_NMF-1), rmss, 'o')

    ax.set_xlabel('Number of Components')
    ax.set_ylabel(r'Average RMSE (m$^{-1}$)')

    ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_explained_variance(
    outfile:str='fig_explained_variance.png',
                 nmf_fit:str='l23'):

    # RMSE
    rmss = []
    for n in range(1,10):
        # load
        d = load_nmf(nmf_fit, N_NMF=n+1)
        N_NMF = d['M'].shape[0]
        recon = np.dot(d['coeff'],
                       d['M'])
        #
        dev = recon - d['spec']
        rms = np.std(dev, axis=1)
        # Average
        avg_rms = np.mean(rms)
        rmss.append(avg_rms)

    # Plot

    fig = plt.figure(figsize=(6,6))
    plt.clf()
    ax = plt.gca()

    ax.plot(2+np.arange(N_NMF-1), rmss, 'o')

    ax.set_xlabel('Number of Components')
    ax.set_ylabel(r'Average RMSE (m$^{-1}$)')

    ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_nmf_basis(outroot:str='fig_nmf_basis',
                 nmf_fit:str='l23', N_NMF:int=4):

    outfile = f'{outroot}_{N_NMF}.png'
    # RMSE
    rmss = []
    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']

    fig = plt.figure(figsize=(12,6))
    plt.clf()
    ax = plt.gca()

    # Plot
    for ss in range(N_NMF):
        ax.step(wave, M[ss], label=r'$\xi_'+f'{ss}'+'$')


    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Basis vector')

    ax.legend()

    #ax.set_yscale('log')
    
    # axes
    plotting.set_fontsize(ax, 15)

    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_fit_cdom(outfile:str='fig_fit_cdom.png',
                 nmf_fit:str='l23', N_NMF:int=4,
                 wv_max:float=600.):

    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']


    # load
    d = load_nmf(nmf_fit, N_NMF=N_NMF)
    M = d['M']
    wave = d['wave']

    if N_NMF==5: 
        ss = 1
    elif N_NMF==4: 
        ss = 0
    else:
        raise IOError("Bad N_NMF")
    a_cdom = M[ss]

    wv_cut = wave < wv_max
    cut_wv = wave[wv_cut]

    # Fit exponentials
    exp_tot_coeff, cov = cdom.fit_exp_tot(wave[wv_cut], 
                                            a_cdom[wv_cut])
    a_cdom_totexp_fit = exp_tot_coeff[0] * cdom.a_exp(
        wave[wv_cut], S_CDOM=exp_tot_coeff[1],
        wave0=exp_tot_coeff[2])
    print(f'Tot exp coeff: {exp_tot_coeff}')
    exp_norm_coeff, cov = cdom.fit_exp_norm(wave[wv_cut], 
                                            a_cdom[wv_cut])
    a_cdom_exp_fit = exp_norm_coeff[0] * cdom.a_exp(wave[wv_cut])

    # Fit power-law
    pow_coeff, pow_cov = cdom.fit_pow(cut_wv, a_cdom[wv_cut])
    a_cdom_pow_fit = pow_coeff[0] * cdom.a_pow(cut_wv, S=pow_coeff[1])

    fig = plt.figure(figsize=(11,5))
    gs = gridspec.GridSpec(1,2)

    # #########################################################
    # Fits as normal
    ax_fits = plt.subplot(gs[0])

    # NMF
    ax_fits.step(wave, M[ss], label=r'$\xi_'+f'{ss}'+'$', color='k')

    ax_fits.plot(cut_wv, a_cdom_exp_fit, 
            color='b', label='CDOM exp', ls='-')
    ax_fits.plot(cut_wv, a_cdom_totexp_fit, 
            color='b', label='CDOM Tot exp', ls='--')
    ax_fits.plot(cut_wv, a_cdom_pow_fit, 
            color='r', label='CDOM pow', ls='-')

    ax_fits.axvline(wv_max, ls='--', color='gray')

    ax_fits.legend()

    # #########################################################
    # CDF
    cdf_NMF = np.cumsum(a_cdom[wv_cut])
    cdf_NMF /= cdf_NMF[-1]
    
    cdf_exp = np.cumsum(a_cdom_exp_fit)
    cdf_exp /= cdf_exp[-1]

    cdf_exptot = np.cumsum(a_cdom_totexp_fit)
    cdf_exptot /= cdf_exptot[-1]

    cdf_pow = np.cumsum(a_cdom_pow_fit)
    cdf_pow /= cdf_pow[-1]

    ax_cdf = plt.subplot(gs[1])

    # Plot
    ax_cdf.step(cut_wv, cdf_NMF, label=r'$\xi_'+f'{ss}'+'$', color='k')
    ax_cdf.plot(cut_wv, cdf_exp, color='b', label='CDOM exp', ls='-')
    ax_cdf.plot(cut_wv, cdf_exptot, color='b', label='CDOM exp', ls='--')
    ax_cdf.plot(cut_wv, cdf_pow, color='r', label='CDOM pow', ls='-')

    # Finish
    for ax in [ax_fits, ax_cdf]:
        plotting.set_fontsize(ax, 15)
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_nmf_coeff(outfile:str='fig_nmf_coeff.png',
                 nmf_fit:str='l23'):

    # load
    d = load_nmf(nmf_fit)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    fig = corner.corner(
        coeff[:,:4], labels=['a0', 'a1', 'a2', 'a3'],
        label_kwargs={'fontsize':17},
        show_titles=True,
        title_kwargs={"fontsize": 12},
        )
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # PCA vs NMF explained variance on L23
    if flg & (2**0):
        fig_l23_pca_nmf_var()

    # L23: PCA and NMF basis functions
    if flg & (2**1):
        fig_nmf_pca_basis()

    # NMF basis
    if flg & (2**11):
        fig_nmf_basis()
        fig_nmf_basis(N_NMF=5)

    # Individual
    if flg & (2**2):
        fig_nmf_indiv()
        fig_nmf_indiv(outfile='fig_nmf_indiv_N5.png',
            N_NMF=5)

    # Coeff
    if flg & (2**3):
        fig_nmf_coeff()

    # Fit CDOM
    if flg & (2**4):
        fig_fit_cdom()

    # Explained variance
    if flg & (2**5):
        fig_explain_variance()

    # NMF RMSE
    if flg & (2**10):
        fig_nmf_rmse()



# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- L23: PCA vs NMF Explained variance
        #flg += 2 ** 1  # 2 -- L23: PCA and NMF basis

        #flg += 2 ** 0  # 1 -- RMSE

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
    else:
        flg = sys.argv[1]

    main(flg)