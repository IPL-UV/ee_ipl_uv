'''
Created on May 18, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
import ee
from __builtin__ import Exception


class PercentileFilter:
    """Simple class which implements a simple percentile filter"""

    def __init__(self, threshold=1.645, percentile=50,
                 bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
                 filter_percentiles=[0, 100]):
        self.threshold = threshold
        self.percentile = percentile
        self.bands = bands
        self.filter_percentiles = filter_percentiles
        self.percentile_image = None
        self.sd_from_percentile_image = None

    def train(self, imgCollection):
        """Given an imgCollection, it computes the percentile and the mean absolute deviation from 
           such percentile filtering values defined by self.filter_percentiles """
        bandsImgCollection = imgCollection.select(self.bands)
        percentiles_compute = [self.percentile];
        if self.filter_percentiles[0] > 0:
            percentiles_compute.append(self.filter_percentiles[0])
        if self.filter_percentiles[1] < 100:
            percentiles_compute.append(self.filter_percentiles[1])

        # Compute the percentiles
        percentiles_image = bandsImgCollection.reduce(ee.Reducer.percentile(percentiles_compute))
        # This will produce an image with len(percentiles_compute)*len(self.bands) bands

        # We keep the percentile self.percentile to contain only the bands named Bi_p50
        band_names_percentile = [bi + "_p" + str(self.percentile) for bi in self.bands]
        self.percentile_image = percentiles_image.select(band_names_percentile)

        # Compute the matrix sd_from_percentile_image (X_dev)       
        if self.filter_percentiles[0] > 0:
            band_names_percentile_low = [bi + "_p" + str(self.filter_percentiles[0]) for bi in self.bands]
            low_percentile = percentiles_image.select(band_names_percentile_low)
        if self.filter_percentiles[1] < 100:
            band_names_percentile_high = [bi + "_p" + str(self.filter_percentiles[1]) for bi in self.bands]
            high_percentile = percentiles_image.select(band_names_percentile_high)

        # Computes residuals putting 0 if the values are >high_percentile or < low_percentile
        def substract_with_mask(img):
            resta = img.subtract(self.percentile_image)
            if self.filter_percentiles[0] > 0:
                resta = resta.where(img.lt(low_percentile), 0)
            if self.filter_percentiles[1] < 100:
                resta = resta.where(img.gt(high_percentile), 0)

            return resta.abs()

        # imgCollection of residuals            
        squared_diferences_collection = bandsImgCollection.map(substract_with_mask)

        # Reduce imgCollection to img and we keep X_dev on sd_from_percentile_image field of the object
        self.sd_from_percentile_image = squared_diferences_collection.mean()

    def predict(self, image, apply_threshold=True, cloud_and_shadows=False):
        """Given an image, it returns a mask image with 1 for potentially cloudy pixels. 
        If cloud_and_shadows flag is true returns 1 for shadows 2 for clouds """

        if self.percentile_image is None:
            raise AssertionError("Model has not been trained. Call train function before predict")

        bandImage = image.select(self.bands)
        # Calculate multiband  squared residuals
        image_normalized = bandImage.subtract(self.percentile_image).divide(self.sd_from_percentile_image)

        if cloud_and_shadows:
            # Compute single band mean of residuals (reduce on band dimension)
            signed_mean_positive = image_normalized.reduce(ee.Reducer.mean()).gt(0)

        image_normalized = image_normalized.pow(2)

        # Calculate the sigle band norm of the residuals (reduce on band dimension)
        single_band_image_residuals = image_normalized.reduce(ee.Reducer.mean()).sqrt()

        if apply_threshold:
            mask_cloud_and_shadows = single_band_image_residuals.gt(self.threshold)
            if cloud_and_shadows:
                return mask_cloud_and_shadows.where(signed_mean_positive.And(mask_cloud_and_shadows), 2).uint16()

            return mask_cloud_and_shadows

        return single_band_image_residuals
