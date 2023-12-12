""" Methods for I/O of CNMF """

import os
from importlib import resources

import numpy as np

def nmf_filename(nmf_fit:str, N_NMF:int, iop:str=None):
    path = os.path.join(resources.files('cnmf'),
                        'data', nmf_fit)
    outroot = os.path.join(path, f'{nmf_fit}_NMF_{N_NMF:02d}')
    if iop is not None:
        outroot += f'_{iop}'
    # FInish
    nmf_file = outroot+'.npz'
    return nmf_file

def save_nmf(outfile:str, M:np.ndarray, coeff:np.ndarray, spec:np.ndarray,
    mask:np.ndarray, err:np.ndarray, wave:np.ndarray,
    Rs:np.ndarray):
    """
    Save the NMF results to a file in npz format.

    Args:
        outfile (str): The path of the output file.
        M (np.ndarray): The spatial components matrix.
        coeff (np.ndarray): The temporal coefficients matrix.
        spec (np.ndarray): The spectral components matrix.
        mask (np.ndarray): The binary mask matrix.
        err (np.ndarray): The error matrix.
        wave (np.ndarray): The wavelength array.
        Rs (np.ndarray): The spatial correlation matrix.
    """
    # Create output directory?
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    # Save
    np.savez(outfile, M=M, coeff=coeff,
             spec=spec, mask=mask,
             err=err, wave=wave, Rs=Rs)
    print(f'Wrote: {outfile}')
             

def load_nmf(nmf_fit:str, N_NMF:int, iop:str=None):
    """ Load a NMF model

    Args:
        nmf_fit (str): _description_
        N_NMF (int, optional): _description_. Defaults to None.
        iop (str, optional): _description_. Defaults to 'a'.

    Raises:
        IOError: _description_

    Returns:
        dict-like: numpy save object
    """

    # File name
    nmf_file = nmf_filename(nmf_fit, N_NMF, iop=iop)

    # Load + Return
    return np.load(nmf_file)
