""" This module contains functions for statistical analysis of NMF results """

import os
from importlib import resources

import numpy as np
import pandas
from matplotlib import pyplot as plt
from importlib import resources

from oceancolor.iop import cross


def evar_nmf(v_target:np.array, w:np.array, h:np.array):
    v_est = np.dot(w, h)
    rss = np.sum(np.square(v_est - v_target))
    evar = 1 - rss / np.sum(np.square(v_est))
    return (rss, evar)

def evar_computation(nmf_fit:str, N_NMF:int=None, iop:str='a'):
    d_npz = load_nmf(nmf_fit, N_NMF, iop)
    v_target = d_npz['spec']
    #######################################
    h = d_npz['M']
    w = d_npz['coeff']
    # we think it should be, but given data
    # requires above code
    # h = d_npz['coeff']
    # w = d_npz['M']
    ######################################
    rss, evar = evar_nmf(v_target, w, h)
    return (rss, evar)

def evar_for_all(save_path, iop:str='a'):
    print("Computation Starts.")
    evar_list, index_list = [], []
    for i in range(2, 11):
        _, evar_i = evar_computation("L23", i, iop)
        evar_list.append(evar_i)
        index_list.append(i)
    result_dict = {
        "index_list": index_list,
        "exp_var": evar_list,
    }
    df_exp_var = pandas.DataFrame(result_dict)
    df_exp_var.set_index("index_list", inplace=True)
    df_exp_var.to_csv(save_path, header=False)    
    print("Computation Ends Successfully!")

def evar_plot(save_path, iop:str='a'):
    print("Computation Starts.")
    evar_list, index_list = [], []
    for i in range(2, 11):
        _, evar_i = evar_computation("L23", i, iop)
        evar_list.append(evar_i)
        index_list.append(i)
    plt.figure(figsize=(10, 8))
    plt.plot(index_list, evar_list, '-o', color='blue')
    plt.axhline(y = 1.0, color ="red", linestyle ="--") 
    plt.xlabel("Dim of Feature space", fontsize=15)
    plt.ylabel("Explained Variance", fontsize=15)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("Plot Ends Successfully!")