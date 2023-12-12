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
from cnmf.oceanography import iops
from cnmf import zhu_nmf as nmf

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

    # Seaborn
    sns.set(style="whitegrid")
    sns.set_palette("pastel")
    #sns.set_palette("husl")

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
            # Step plot
            sns.lineplot(x=wave, y=M[ii]/nrm, 
                         label=f'{itype}:'+r'  $\xi_'+f'{ii+1}'+'$',
                         ax=ax, lw=2)#, drawstyle='steps-pre')
            #ax.step(wave, M[ii]/nrm, label=f'{itype}:'+r'  $\xi_'+f'{ii+1}'+'$')

        ax.set_xlabel('Wavelength (nm)')

        lbl = 'PCA' if ss == 0 else 'NMF'
        ax.set_ylabel(lbl+' Basis Functions')

        loc = 'upper right' if ss == 1 else 'upper left'
        ax.legend(fontsize=15, loc=loc)


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

def fig_l23_fit_nmf(outfile:str='fig_l23_fit_nmf.png',
                 nmf_fit:str='L23', N_NMF:int=4,
                 icdom:int=1,
                 cdom_max:float=600.):

    # Load
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, 'a')
    M = d['M']
    wave = d['wave']
    a_cdom = M[icdom]

    # #########################################################
    # CDOM
    wv_cut = wave < cdom_max
    cut_wv = wave[wv_cut]

    # Fit exponentials
    exp_tot_coeff, cov = cdom.fit_exp_tot(
        wave[wv_cut], a_cdom[wv_cut])
    a_cdom_totexp_fit = exp_tot_coeff[0] * cdom.a_exp(
        wave[wv_cut], S_CDOM=exp_tot_coeff[1])
        #wave0=exp_tot_coeff[2])
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
    # CDOM fits
    ax_cdom = plt.subplot(gs[0])

    # NMF
    ax_cdom.step(wave, M[icdom], 
                 label=r'$\xi_'+f'{icdom}'+'$', color='k',
                 lw=2)

    #ax_cdom.plot(cut_wv, a_cdom_exp_fit, 
    #        color='b', label='CDOM exp', ls='-')
    ax_cdom.plot(cut_wv, a_cdom_totexp_fit, 
            color='cyan', 
            label=r'Exponential ($S='+f'{exp_tot_coeff[1]:0.3f}'+r'$)', 
            ls='--', lw=2)
    ax_cdom.plot(cut_wv, a_cdom_pow_fit, 
            color='r', label='Power Law '+r'($\alpha='+f'{pow_coeff[1]:0.1f}'+r'$)', 
            ls=':', lw=2)

    ax_cdom.axvline(cdom_max, ls='--', color='gray')

    ax_cdom.legend(fontsize=15.)

    # Label the axes
    ax_cdom.set_xlabel('Wavelength (nm)')
    ax_cdom.set_ylabel(r'Absorption Coefficient (m$^{-1}$)')

    # #########################
    # Fit the chlorophyll

    # Finish
    for ax in [ax_cdom]:#, ax_cdf]:
        plotting.set_fontsize(ax, 16)
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

# #########################################################
# #########################################################
def fig_l23_tara_a_contours(
    outfile:str='fig_l23_tara_a_contours.png',
    nmf_fit:str='L23',
    N_NMF:int=4, iop:str='a'):

    sns.set(style="whitegrid")

    # Load L23 fit
    d = cnmf_io.load_nmf(nmf_fit, N_NMF, iop)
    M = d['M']
    coeff = d['coeff']
    wave = d['wave']

    # Load Tara
    d_tara = cnmf_io.load_nmf('Tara_L23', N_NMF, iop)
    tara_coeff = d_tara['coeff']

    # Scale?
    scale = True
    if scale:
        for ss in range(4):
            med_l23 = np.median(coeff[ss,:])
            med_tara = np.median(tara_coeff[ss,:])
            print(f"Scale: {med_l23} {med_tara} {med_l23/med_tara}")
            #
            tara_coeff[ss,:] *= med_l23/med_tara


    # #########################################################
    # Figure
    figsize=(6,6)
    fig = plt.figure(figsize=figsize)
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax= plt.subplot(gs[0])

    # #########################################################
    # L23 Contours plot
    sns.kdeplot(
        x=coeff[:,0], 
        y=coeff[:,1],
        ax=ax,
        kind='kde', label='L23')

    # #########################################################
    # Tara Contours plot
    sns.kdeplot(
        x=tara_coeff[:,0], 
        y=tara_coeff[:,1],
        color='r',
        ax=ax,
        kind='kde', label='Tara', ls=':')

    # Finish
    ax.set_xlabel(r'$a_1$')
    ax.set_ylabel(r'$a_2$')

    ax.set_xlim(0., 0.02)
    ax.set_ylim(0., 0.04)

    ax.legend(fontsize=15.)

    # Finish
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")

def fig_l23_a_corner(
    outfile:str='fig_l23_a_corner.png',
    nmf_fit:str='L23'):

    # Load
    d = cnmf_io.load_nmf(nmf_fit, 4, 'a')
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

    # L23: Fit NMF 1, 2
    if flg & (2**2):
        fig_l23_fit_nmf()

    # L23: a1, z2 contours
    if flg & (2**3):
        fig_l23_tara_a_contours()


    # NMF basis
    if flg & (2**11):
        fig_nmf_basis()
        fig_nmf_basis(N_NMF=5)

    # Individual
    if flg & (2**12):
        fig_nmf_indiv()
        fig_nmf_indiv(outfile='fig_nmf_indiv_N5.png',
            N_NMF=5)

    # Coeff
    if flg & (2**14):
        fig_l23_a_corner()

    # Fit CDOM
    if flg & (2**15):
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
        #flg += 2 ** 2  # 4 -- L23: Fit NMF 1, 2

        #flg += 2 ** 3  # 8 -- L23+Tara; a1, a2 contours

        #flg += 2 ** 0  # 1 -- RMSE

        #flg += 2 ** 2  # 4 -- Indiv
        #flg += 2 ** 3  # 8 -- Coeff
        #flg += 2 ** 4  # 16 -- Fit CDOM
        #flg += 2 ** 5  # 32 -- Explained variance
        
        flg += 2 ** 14  # 32 -- Explained variance
    else:
        flg = sys.argv[1]

    main(flg)