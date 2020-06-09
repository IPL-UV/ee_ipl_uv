'''
Created on June 02, 2016

@author: Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es
'''
import ee
import numpy as np
import pandas as pd
from datetime import datetime
from ee_ipl_uv import file_utils
import os
import requests
import logging
import shutil
import time

logger = logging.getLogger(__name__)


def ExtractColumnAseeArray1D(ftcol, column):
    """
    Extract a column of a feature collection as an ee.Array object

    :param ftcol:
    :type ftcol: ee.FeatureCollection
    :param column: str with the name of the column
    :type column: str
    :return:
    """
    num_rows = ee.Number(ftcol.size())
    return ee.Array(ftcol.toList(num_rows).map(lambda feat: ee.Feature(feat).get(column)))


def eeFeatureCollectionToeeArray(ftcol, columns):
    """ (ee.FeatureCollection, List[str], int) -> ee.Array

    :param ftcol:
    :type ftcol: ee.FeatureCollection.
    :param columns: List[str] colums from the ftcol to retrieve
    :return: ee.Array
    """
    num_rows = ee.Number(ftcol.size())

    bands_to_predict_as_array = ftcol.map(
        lambda ftr: ee.Feature(None, {"lista": ftr.toArray(columns).toList()}))

    def extractFromFeature(ftr):
        ftr = ee.Feature(ftr)
        return ftr.get("lista")

    return ee.Array(bands_to_predict_as_array.toList(num_rows).map(extractFromFeature))


def eeFeatureCollectionToNumpy(ftcol, columns):
    """ (ee.FeatureCollection, List[str], int) -> numpy.array

    :return: numpy.array
    """
    ee_array = eeFeatureCollectionToeeArray(ftcol, columns)
    return np.asanyarray(ee_array.getInfo())


def eeFeatureCollectionToPandas(feature_col, properties=None, with_task=False,
                                filename=None, mounted_drive=False):
    """
    Converts ee.FeatureCollection server obj to pandas.DataFrame local obj.

    :param feature_col: feature collection to export
    :type feature_col: ee.FeatureCollection
    :param properties: (optional) list of columns to export
    :param with_task: (default false). If download is done throw ee.batch.Export.table
    :param filename: (optional) if None csv downloaded will be removed.
    :param mounted_drive: if drive is mounted we don't need to use pyDrive
    If present nothing will be downloaded

    :return: pandas.DataFrame object
    """
    # Features is a list of dict with the output
    remove_file = not mounted_drive
    filetype = "csv"
    if filename is None:
        filename = file_utils.createTempFile(params={"format": filetype},
                                             prefix="pandas_ftcol")
    else:
        filename_full = file_utils.addFormat(filename, filetype)
        if os.path.isfile(filename_full):
            return _readCSV(filename_full)

        remove_file = False

    prefix = file_utils.removeFormat(os.path.basename(filename), "csv")

    if with_task:
        from ee_ipl_uv.download import WaitAndDownload, WaitTask
        if properties is not None:
            feature_col = feature_col.select(properties)

        tarea = ee.batch.Export.table.toDrive(feature_col,
                                              prefix,
                                              folder="ee_ipl_uv_downloads",
                                              fileFormat=filetype)
        tarea.start()
        if mounted_drive:
            WaitTask(tarea)
            filename = os.path.join("/content/drive/My Drive/ee_ipl_uv_downloads/", prefix+".csv")
            if not os.path.exists(filename):
                logger.info("File %s not ready in drive. Waiting 30 seconds" % filename)
                time.sleep(30)
            assert os.path.exists(filename), "%s does not exists in the drive" % filename
        else:
            filename = WaitAndDownload(tarea, prefix,
                                       formato="csv", force=True)
    else:
        if properties is None:
            url = feature_col.getDownloadURL(filetype=filetype)
        else:
            properties_list = properties.getInfo()
            url = feature_col.getDownloadURL(filetype=filetype,
                                             selectors=properties_list)

        logger.debug("Downloading data from: " + url)

        r_link = requests.get(url, stream=True)
        if r_link.status_code == 200:
            with open(filename, 'wb') as f:
                r_link.raw.decode_content = True
                shutil.copyfileobj(r_link.raw, f)

    logger.debug("File downloaded, reading csv: " +filename)
    datos = _readCSV(filename)

    if remove_file:
        os.remove(filename)

    return datos


def _readCSV(csv_f):
    datos = pd.read_csv(csv_f, index_col=False)

    # Convert system:time_start, system:time_end to datetime
    columns_time = ["system:time_start", "system:time_end"]
    columns_time_present = filter(lambda col: col in columns_time, datos.columns)
    for col_p in columns_time_present:
        datos[col_p] = datos[col_p].apply(lambda x: datetime.utcfromtimestamp(x / 1000))

    return datos


def eeImageCollectionToPandas(img_col, properties=None):
    """ Converts image collection metadata to pandas.DataFrame"""
    if properties is None:
        properties = ee.Image(img_col.first()).propertyNames()
    else:
        properties = ee.List(properties)

    def extractFeatures(img):
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        return ee.Feature(None, dictio)

    featureCol = ee.FeatureCollection(img_col.map(extractFeatures))
    return eeFeatureCollectionToPandas(featureCol, properties=None)

