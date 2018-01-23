'''
Created on June 15, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
import ee


def ComputeNormalizationFeatureCollection(feature_collection,
                                          bands_to_normalize,
                                          only_center_data=False,
                                          weight=None):
    """
    Compute normalization: returns the feature_collection normalized together with the mean
    and std.

    >>> ee.Initialize()

    >>> im = ee.Image('LANDSAT/LC8_L1T_TOA/LC81980332015119LGN00').select(["B2","B3"])
    >>> ftcol = im.sample(numPixels=10,seed=23)
    >>> ftcol_norm, mean, std = ComputeNormalizationFeatureCollection(ftcol,["B2","B3"])

    :param feature_collection:
    :param bands_to_normalize:
    :param only_center_data: if
    :param weight: weight column
    :return: feature_collection normalized, dictionary with mean of each band, dict with
    sd of each band
    :rtype ee.FeatureCollection, ee.Dictionary, ee.Dictionary
    """
    bands_to_normalize_server = ee.List(bands_to_normalize)
    if weight is not None:
        weights = [weight for b in bands_to_normalize]
        mean = feature_collection.reduceColumns(
            # reducer=ee.Reducer.mean().repeat(ee.Number(bands_to_normalize.size())),
            reducer=ee.Reducer.mean().forEach(bands_to_normalize_server),
            selectors=bands_to_normalize_server,
            weightSelectors=weights
        )
    else:
        mean = feature_collection.reduceColumns(
            # reducer=ee.Reducer.mean().repeat(ee.Number(bands_to_normalize.size())),
            reducer=ee.Reducer.mean().forEach(bands_to_normalize_server),
            selectors=bands_to_normalize_server
        )

    if not only_center_data:
        sd = feature_collection.reduceColumns(
            reducer=ee.Reducer.stdDev().forEach(bands_to_normalize_server),
            selectors=bands_to_normalize_server)
    else:
        sd = None

    def NormalizeFeature(feature):
        feature = ee.Feature(feature)

        if not only_center_data:
            values = bands_to_normalize_server.map(
                lambda bn: ee.Number(feature.get(bn)).subtract(mean.get(bn)).divide(sd.get(bn)))
        else:
            values = bands_to_normalize_server.map(
                lambda bn: ee.Number(feature.get(bn)).subtract(mean.get(bn)))

        dictio = ee.Dictionary.fromLists(bands_to_normalize_server, values)
        return feature.setMulti(dictio)

    return feature_collection.map(NormalizeFeature), mean, sd


def ApplyToFeature(properties, function):
    properties = ee.List(properties)

    def funcion_apply(feature_iter):
        feature_iter = ee.Feature(feature_iter)
        values = properties.map(lambda bn: function(ee.Number(feature_iter.get(bn))))
        dictio = ee.Dictionary.fromLists(properties, values)
        return feature_iter.setMulti(dictio)
    return funcion_apply


def ApplyNormalizationImage(img, bands, mean, sd):
    for band in bands:
        im_copy = img.select(band)
        if sd is not None:
            im_copy = im_copy.subtract(ee.Number(mean.get(band))).divide(ee.Number(sd.get(band)))
        else:
            im_copy = im_copy.subtract(ee.Number(mean.get(band)))
        img = img.addBands(im_copy, overwrite=True)
    return img


def ApplyDenormalizationImage(img, bands, mean, sd=None):
    for band in bands:
        im_copy = img.select(band)
        if sd is not None:
            im_copy = im_copy.multiply(ee.Number(sd.get(band))).add(ee.Number(mean.get(band)))
        else:
            im_copy = im_copy.add(ee.Number(mean.get(band)))

        img = img.addBands(im_copy, overwrite=True)
    return img
