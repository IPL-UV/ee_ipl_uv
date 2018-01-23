'''
Created on May 16, 2016

@author: Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es
'''
import ee


def GenerateBandNames(bands, sufix):
    """
    Concat to each element in bands list the sufix string

    >>> import ee
    >>> ee.Initialize()
    >>> GenerateBandNames(ee.List(["B1","B2"]), "_lag_1").getInfo()
    [u'B1_lag_1', u'B2_lag_1']

    >>> import ee
    >>> ee.Initialize()
    >>> bands=ee.Image('LANDSAT/LC8_L1T_TOA_FMASK/LC81980332015119LGN00').bandNames()
    >>> GenerateBandNames(bands, "_lag_1").getInfo()
    [u'B1_lag_1', u'B2_lag_1', u'B3_lag_1', u'B4_lag_1', u'B5_lag_1', u'B6_lag_1', u'B7_lag_1', u'B8_lag_1', u'B9_lag_1', u'B10_lag_1', u'B11_lag_1', u'BQA_lag_1', u'fmask_lag_1']


    :param bands: ee.List where each Element is a ee.String
    :param sufix: str to concat
    :return: list
    :rtype ee.List
    """
    bands = ee.List(bands)
    return bands.map(lambda band: ee.String(band).cat(ee.String(sufix)))


def GenerateBandLags(bands, sufixes):
    """ Return cartesian product of bands and sufixes concat"""
    return sufixes.map(lambda sf: GenerateBandNames(bands, sf)).flatten()


def AddAllLags(imgCollection, num_lags=1, filtering_property=None):
    """
    Given a imgCollection it returns a list with the same number of images of the imageCollection
    where each image contains its bands together with the bands of num_lags images where filtering_property
    does NOT hold
    It also add to the metadata of the image the lagged time stamp ("system:time_start_lag_X")

    :param imgCollection: each img must have system:time_start property
    :param num_lags: number of lagged elements of each image
    :param filtering_property: if None no filtering will be done
    :return: list with images
    :rtype ee.List
    """
    lags_client = range(1, num_lags+1)
    bands_server = ee.Image(imgCollection.first()).bandNames()
    number_of_bands_server = ee.Number(bands_server.size())

    total_number_bands = number_of_bands_server.multiply(num_lags + 1)

    # create zero image with number_of_bands_server x num_lags + 1 bands
    zeros = ee.List.repeat(0, total_number_bands)
    bands_for_select = ee.List.sequence(0, total_number_bands.subtract(1))

    # sufix_lag_server = ["_lag_1",....,"_lag_n-1","_lag_n"]
    sufix_lag_server = ee.List(list(map(lambda lg: "_lag_"+str(lg), lags_client)))
    # sufix_lag_server_initial = ["","_lag_1",....,"_lag_n-1","_lag_n"]
    sufix_lag_server_initial = ee.List([ee.String("")]).cat(sufix_lag_server)
    # sufix_lag_server_shifted = ["","_lag_1",....,"_lag_n-1"]
    sufix_lag_server_shifted = sufix_lag_server_initial.slice(0, num_lags)

    all_bands = GenerateBandLags(bands_server, sufix_lag_server_initial)

    # Important constant otherwise doesn't work
    image_primera = ee.Image.constant(zeros).select(bands_for_select,
                                                    all_bands)

    # Set time property
    name_time_server = sufix_lag_server_initial.map(lambda sufix: ee.String("system:time_start").cat(ee.String(sufix)))
    zeros_time_start_server = ee.List.repeat(0, num_lags+1)
    dictio = ee.Dictionary.fromLists(name_time_server, zeros_time_start_server)
    image_primera = image_primera.set(dictio)
    if filtering_property is not None:
        image_primera = image_primera.set(filtering_property, 1)

    # Create first element for iterate
    lista_ee = ee.List([image_primera])

    def accumulate(image, lista_rec):
        lista_recibida = ee.List(lista_rec)
        previous = ee.Image(lista_recibida.get(-1))
        bands_with_lags_server = GenerateBandLags(bands_server, sufix_lag_server)
        sufix_lag_server_shifted_iteration = sufix_lag_server_shifted

        # if cloud > X bands_with_lags_shifted_server = bands_with_lags_server
        if filtering_property is not None:
            sufix_lag_server_shifted_iteration = ee.Algorithms.If(ee.Number(previous.get(filtering_property)),
                                                                  sufix_lag_server,
                                                                  sufix_lag_server_shifted)
            sufix_lag_server_shifted_iteration = ee.List(sufix_lag_server_shifted_iteration)

        bands_with_lags_shifted_server = GenerateBandLags(bands_server, sufix_lag_server_shifted_iteration)

        previous_add = previous.select(bands_with_lags_shifted_server,
                                       bands_with_lags_server)

        image = image.addBands(previous_add)

        name_time_server_shifted = sufix_lag_server_shifted_iteration.map(
            lambda sufix: ee.String("system:time_start").cat(ee.String(sufix)))

        values_time_server_select = name_time_server_shifted.map(lambda field: previous.get(field))
        name_time_server_select = sufix_lag_server.map(
            lambda sufix: ee.String("system:time_start").cat(ee.String(sufix)))
        dictio_set = ee.Dictionary.fromLists(name_time_server_select, values_time_server_select)
        image = image.set(dictio_set)

        return lista_recibida.add(image)

    lista_retorno = ee.List(imgCollection.iterate(accumulate, lista_ee)).slice(1)
    return lista_retorno

