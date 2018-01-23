import luigi
import os
from ee_ipl_uv import local_image
from ee_ipl_uv import multitemporal_cloud_masking as mcm
import ee
import logging

class RasterTarget(luigi.LocalTarget):
    def __init__(self, path):
        super(RasterTarget, self).__init__(path=path)

    def exists(self):
        return os.path.exists(self.raster_file()) and os.path.exists(self.metadata_file())

    def LocalImage(self):
        return local_image.LocalImage.Load(self.path)

    def raster_file(self):
        return os.path.join(self.path, "raster.tif")

    def metadata_file(self):
        return os.path.join(self.path, "info.json")


class DownloadImageLags(luigi.Task):
    image_index = luigi.Parameter()
    basepath = luigi.Parameter(default="images")
    max_lags = luigi.IntParameter(default=3)
    threshold_cc = luigi.IntParameter(default=8)

    band_names = list(mcm.LANDSAT8_BANDNAMES)

    def output(self):
        return RasterTarget(os.path.join(self.basepath,
                                         self.image_index))

    def load_image(self):
        return ee.Image(self.image_index)

    def run(self):
        if not os.path.exists(self.output().path):
            self.output().makedirs()
            os.mkdir(self.output().path)


        ee.Initialize()
        image = self.load_image()
        prev_images = mcm.SelectImagesTraining(image,
                                               num_images=self.max_lags,
                                               THRESHOLD_CC=self.threshold_cc)
        prev_images = prev_images.toFloat()

        lag_times = ["system:time_start_lag_" + str(x) for x in range(1, self.max_lags + 1)]
        keys_cc = ["CC_lag_" + str(lag) for lag in range(1, self.max_lags + 1)]
        properties = ["system:time_start", "filtering_prop", 'system:index']
        properties += lag_times+keys_cc

        try:
            local_image.ExporteeImage(self.output().path,
                                      prev_images,
                                      properties_ee_img=properties)
        except ee.EEException as e:
            message = str(e)
            if "The value of 'offset' must be non-negative" in message:
                logging.info("The image %s does not have %d previous images\n %s"%(self.image_index,
                                                                                   self.max_lags,
                                                                                   message))
            else:
                raise e


