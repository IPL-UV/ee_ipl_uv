import ee
import re


def clouds_bqa_landsat(ee_img):
    return ee_img.select(['BQA'], ["cloud"]).bitwiseAnd(int('0000000000010000', 2)).gt(0)


class L8L1TImage:
    def __init__(self, index, collection="LANDSAT/LC08/C01/T1_TOA"):
        if collection.endswith("/"):
            collection = collection[:-1]
        self.ee_img = ee.Image(collection+"/"+index)
        self.collection = collection
        self.index = index
        self.clouds_bqa_fun = clouds_bqa_landsat

    def collection_similar(self, region_of_interest=None):
        matches = re.match("LC08_(\d{3})(\d{3})_\d{8}", self.index)
        path, row = matches.groups()
        if region_of_interest is None:
            # region_of_interest = ee.Element.geometry(self.ee_img)
            landsat_collection = ee.ImageCollection(self.collection) \
                .filter(ee.Filter.eq("WRS_ROW", int(row))) \
                .filter(ee.Filter.eq("WRS_PATH", int(path)))
        else:
            landsat_collection = ee.ImageCollection(self.collection) \
                .filterBounds(region_of_interest) \
                .filter(ee.Filter.eq("WRS_ROW", int(row)))

        return landsat_collection

    @classmethod
    def reflectance_bands(cls):
        return ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]

    @classmethod
    def rgb_bands(cls):
        return ["B4", "B3", "B2"]

    @classmethod
    def revisit_time_period(cls):
        return 15

    def all_bands(self):
        return self.reflectance_bands() + ["BQA"]

    def clouds_bqa(self):
        return clouds_bqa_landsat(self.ee_img)


def toa_norm_s2l1c(ee_img):
    ee_img_rads = ee_img.select(S2L1CImage.reflectance_bands()).divide(10000.)
    return ee_img.addBands(ee_img_rads, overwrite=True)


# COPERNICUS/S2/20181107T105231_20181107T105655_T30SYJ
class S2L1CImage:
    def __init__(self, index, collection="COPERNICUS/S2/"):
        if collection.endswith("/"):
            collection = collection[:-1]

        self.index = index
        ee_img = ee.Image(collection+"/"+index)

        ee_img_rads = ee_img.select(S2L1CImage.reflectance_bands()).divide(10000.)
        self.ee_img = ee_img.addBands(ee_img_rads, overwrite=True)
        self.collection = collection
        self.clouds_bqa_fun = clouds_bqa_sentinel2
        self.toa_norm_fun = toa_norm_s2l1c

    @classmethod
    def reflectance_bands(cls):
        return ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

    @classmethod
    def rgb_bands(cls):
        return ["B4", "B3", "B2"]

    @classmethod
    def revisit_time_period(cls):
        return 7

    def collection_similar(self, region_of_interest=None):
        s2collection = ee.ImageCollection(self.collection)
        if region_of_interest is not None:
            s2collection = s2collection.filterBounds(region_of_interest)

        # Force same tile
        mgrs_tile = self.index.split("_")[-1][1:]
        assert len(mgrs_tile) == 5, "Unrecognized index mrgs tile {} {}".format(self.index, mgrs_tile)
        s2collection = s2collection.filter(ee.Filter.eq("MGRS_TILE", mgrs_tile))
        return s2collection

    def all_bands(self):
        return self.reflectance_bands() + ["QA60"]

    def clouds_bqa(self):
        return clouds_bqa_sentinel2(self.ee_img)


def clouds_bqa_sentinel2(ee_img):
    """
    var qa = image.select('QA60');
    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
                 .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    :return:
    """
    qa = ee_img.select(['QA60'], ["cloud"])
    cloud_bit_mask = int("0000010000000000", 2)
    cirrus_bit_mask = int("0000100000000000", 2)
    return qa.bitwiseAnd(cloud_bit_mask).gt(0).Or(qa.bitwiseAnd(cirrus_bit_mask).gt(0))


