"""
Script to reproduce the results of the paper Multitemporal Cloud Masking in the Google Earth Engine (2018).

The following command line downloads one image patch with the cloud mask computed with our method and its corresponding ground truth from the Biome dataset

python reproducibility.py DownloadImageResults --image-index LC80290372013257LGN00 --split 013_011 --method percentile --basepath /folder/to/download/patch

To download all the patches:

python reproducibility.py DownloadAll --basepath /folder/to/download/patches

Extra dependencies (apart from package ee_ipl_uv):
 - luigi
 - pydrive

"""
import luigi
import ee
import ee_ipl_uv.luigi_utils
from ee_ipl_uv import multitemporal_cloud_masking
import requests
import os
import json

def get_location_splits():
    filename = "locations_splits.json"
    if not os.path.exists(filename):
        r = requests.get(url="http://isp.uv.es/projects/cdc/GEE_CLOUDS/locations_splits.json")
        locations = r.json()
        with open(filename,"w") as f:
            json.dump(locations,f)
    else:
        with open(filename,"r") as f:
            locations = json.load(f)

    return locations

class DownloadImageResults(ee_ipl_uv.luigi_utils.DownloadImage):
    split = luigi.Parameter()
    method = luigi.ChoiceParameter(choices=["percentile"],var_type=str,
                                   default="percentile")

    def output(self):
        return ee_ipl_uv.luigi_utils.RasterTarget(os.path.join(self.basepath,
                                                               self.image_index+"_"+self.split+"_"+self.method))
    def load_region_of_interest(self):
        locations = get_location_splits()
        return [[p[1], p[0]] for p in locations[str(self.image_index)][str(self.split)][0]]

    def load_image(self):
        image_predict_clouds = ee.Image('LANDSAT/LC8_L1T_TOA_FMASK/' + str(self.image_index))

        # Select region of interest (lng,lat)
        pol = self.load_region_of_interest()
        region_of_interest = ee.Geometry.Polygon(pol)

        cloud_score_percentile, pred_percentile = multitemporal_cloud_masking.CloudClusterScore(image_predict_clouds,
                                                                                                region_of_interest,
                                                                                                method_pred=self.method,
                                                                                                num_images=3)

        ground_truth = ee.Image("users/gonzmg88/LANDSAT8_CLOUDS/" + self.image_index + "_fixedmask")

        image_download = image_predict_clouds.addBands(cloud_score_percentile.select(["cluster"], ["cloudscore"])) \
            .addBands(ground_truth.select(["b1"], ["fixedmask"]))\
            .addBands(pred_percentile).clip(region_of_interest).toFloat()

        properties = ["system:time_start", 'system:index']

        return image_download, properties


class DownloadAll(luigi.WrapperTask):
    basepath = luigi.Parameter(default="images_gee")
    method = luigi.ChoiceParameter(choices=["percentile"],
                                   default="percentile",var_type=str)

    def requires(self):
        locations = get_location_splits()
        tareas = []
        for index_name,v in locations.items():
            for split_name,pol in v.items():
                tarea = DownloadImageResults(image_index=index_name,
                                             basepath=self.basepath,
                                             method=self.method,
                                             split=split_name)
                tareas.append(tarea)
        return tareas

if __name__ == "__main__":
    luigi.run(local_scheduler=True)

