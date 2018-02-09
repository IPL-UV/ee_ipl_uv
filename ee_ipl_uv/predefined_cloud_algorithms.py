'''
Created on May 18, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
import numpy as np


def QACloudMaskLocal(d2_array):
    """
    Obtain clod mask from np.array

    :param d2_array:
    :return:
    """
    d2_array = d2_array.astype(np.uint16)
    pattern = np.uint16(int('1010000000000000', 2))
    return np.bitwise_and(d2_array,pattern) > 0


def FMaskLandsatLocal(d2_array):
    """
    Return clouds and shadows from mask: https://code.earthengine.google.com/dataset/LANDSAT/LC8_L1T_TOA_FMASK
    :param d2_array:
    :return:

    fmask: cloud mask
        0=clear
        1=water
        2=shadow
        3=snow
        4=cloud

    """
    d2_array = d2_array.astype(np.uint16)
    # Get only clouds and shadows https://code.earthengine.google.com/dataset/LANDSAT/LC8_L1T_TOA_FMASK
    mascara_fmask = np.where(d2_array == 4, 2, 0)
    mascara_fmask = np.where(d2_array == 2, 1, mascara_fmask)
    #return np.ma.MaskedArray(mascara_fmask, mascara_fmask == 0)
    return mascara_fmask
    
    
    
    