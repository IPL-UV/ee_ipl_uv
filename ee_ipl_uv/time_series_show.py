'''
Created on June 15, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
from ee_ipl_uv import download
from datetime import datetime
import os


def LaggedThumbsImages(image, lags,  params={}, bands=None,
                       add_timestamp=True):
    if bands is None:
        bands = image.bandNames().getInfo()

    params["bands"] = ",".join(bands)

    images_files = []
    for lag in lags:
        if lag != 0:
            sufix = "_lag_"+str(lag)
        else:
            sufix = ""
        bands_down = map(lambda x: x+sufix, bands)
        params["bands"] = ",".join(bands_down)
        timestamp_it = None
        if add_timestamp:
            timestamp_it = datetime.utcfromtimestamp(image.get("system:time_start"+sufix).getInfo() / 1000) \
                .strftime("%Y-%m-%d %H:%M:%S")
        first_img = download.MaybeDownloadThumb(image,
                                                params=params,
                                                footprint=timestamp_it)
        images_files.append(first_img)

    images_files.reverse()
    return images_files


def ShowLaggedThumbs(image, lags,  params={}, bands=None, image_name=None,
                     add_timestamp=True):

    images_files = LaggedThumbsImages(image, lags, params, bands, add_timestamp)

    mosaico = download.MosaicImageList(images_files, [1, len(lags)], image_name)
    for image in images_files:
        os.remove(image)

    return mosaico
