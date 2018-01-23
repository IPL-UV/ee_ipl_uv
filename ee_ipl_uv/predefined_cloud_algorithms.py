'''
Created on May 18, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
import ee
import numpy as np


def QACloudMask(img, strict=True):
    """
    Band explanation. If strict is False it will return clouds with
    confidence >33 otherwise >66. https://landsat.usgs.gov/collectionqualityband
    """
    # band_names = img.bandNames().getInfo()
    # if 'BQA' not in band_names:
    #    raise AssertionError("image do not contains BQA mask")
    # Maybe cloud:
    # pattern = Math.pow(2,15)+Math.pow(2,14)+Math.pow(2,13)+Math.pow(2,12)
    
    #if strict:
    #    pattern = int('1111000000000000',2)
    #    pattern_cloud = int('1100000000000000',2)
    #    pattern_cirrus = int('0011000000000000',2)
    #    return img.select('BQA').bitwiseAnd(pattern).eq([pattern,pattern_cloud,pattern_cirrus]).reduce(ee.Reducer.sum()).gt(0)
    
    #pattern = int('1010000000000000',2)

    return img.select('BQA').bitwiseAnd(int("10000", 2)).gt(0)
    
    
def FMask(im, cloud=None, shadow=None, kelvin_termal_band='B10'):
    """
    img shoud be an unclipped image.
    cloud and shadows are basic cloud and shadows mask that requires the algorithm.
    """
    if cloud is None:
        cs = ee.Algorithms.Landsat.simpleCloudScore(im)
        cloud = cs.select('cloud').gte(50.0)
        cloud = cloud.updateMask(cloud)
    # ee.mapclient.addToMap(cloud, {'palette':'ff0000'}, 'clouds')
    
    # Threshold B6 to get dark pixels 
    # Very likely NOT the right way to do this ...
    if shadow is None:
        shadow = im.select('B6').lt(0.15)
        shadow = shadow.updateMask(shadow)
    # ee.mapclient.addToMap(shadow, {'palette':'ffff00'}, 'shadows')
    
    # Get thermal band in deg C
    thermal = im.select(kelvin_termal_band).subtract(273.15)
    # ee.mapclient.addToMap(thermal, {'min':-10, 'max':30}, 'thermal')
    
    # Calculate the 17.5 and 82.5 percentiles from this image
    thresholds = thermal.reduceRegion(
      reducer= ee.Reducer.percentile([17.5, 82.5]),
      bestEffort=True
    )
    
    p_17_5 = thresholds.get('B10_p18')
    p_82_5 = thresholds.get('B10_p83')
    
    # print(p_17_5)
    # print(p_82_5)
    
    # Calculate FMask cloud masking
    # 1 (yellow) - shadows
    # 2 (green) - clouds
    fmask = ee.Algorithms.FMask.matchClouds(im, cloud, shadow,
                                            thermal, p_17_5, p_82_5)
    return fmask

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
    
    
    
    