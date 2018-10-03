import json
from datetime import datetime
import ee
from ee_ipl_uv import download
import os
import random


RASTER_NAME = "raster"


class LocalImage:
    """
    Class to load downloaded data from ee. It expects some reader and 
    some metadata. 
    """
    def __init__(self,  tiff, metadata={}):
        from skimage.external import tifffile
        self.tiff = tiff
        self.metadata = metadata
        self.memmap = tifffile.imread(tiff, memmap=True) #out="memmap"
        self.nrows = self.memmap.shape[0]
        self.ncols = self.memmap.shape[1]

        self.nombre = os.path.splitext(os.path.basename(self.tiff))[0]

    @staticmethod
    def Load(local_dir):
        """
        Method to load an image from a directory. It is expected to work with 
        ExporteeImage function. It loads a reader from RASTER_NAME+".tif" and
        metadata from info.json 
        """
        # Load band names

        info_file = os.path.join(local_dir,"info.json")
        info = {}
        if os.path.exists(info_file):
            with open(info_file, "r") as infile:
                info = json.load(infile)

        tiff = os.path.join(local_dir, RASTER_NAME + ".tif")

        return LocalImage(tiff, info)

    def eeImage(self):
        """
        Method to obtain back the server image object.

        :return: server ee.Image object
        """
        return ee.Image(self.metadata["image_identity"])

    def date(self):
        """
        time_start attribute of the image in datetime format

        :return: datetime object.
        """
        if "system:time_start" not in self.metadata:
            img = self.eeImage()
            self.metadata["system:time_start"] = img.get("system:time_start").getInfo()

        return datetime.utcfromtimestamp(self.metadata["system:time_start"]/1000)

    def readBands(self):
        """
        Read the image as a np.array in format (rows,cols,channels)

        :return: np.ndarray with image in format (rows,cols,channels)
        """
        return self.memmap

    def bandNames(self):
        """
        Return name of the bands

        :return: array with name of bands
        """
        return self.metadata["bands"]

    def saveMetadata(self, local_dir):
        """
        save the metadata of the object into the file os.path.join(local_dir,"info.json")

        :param local_dir:
        """
        with open(os.path.join(local_dir,"info.json"), 
                  "w") as outfile:
            json.dump(self.metadata, outfile)


def ExporteeImage(local_dir, ee_img,region=None, metadata={},
                  properties_ee_img=None):
    """
    Exports an ee.Image with a ee.task into the folder local_dir.
    It saves within that dir a tif file called RASTER_NAME and a json file
    called info.json with the metadata.
    
    :param local_dir - dir where the ee_img is stored.
    :param ee_img - ee.Image object to download
    :param metadata - metadata of the image
    :param properties_ee_img - list with properties to extract from the image
     with ee.Image.get method that will be stored in the json file.
    
    """
    
    if properties_ee_img is not None:
        properties_serv = ee.List(properties_ee_img)
        values = properties_serv.map(lambda prop: ee_img.get(prop))
        info = ee.Dictionary.fromLists(properties_ee_img, values)
        info = info.set("bands",ee_img.bandNames()).getInfo()        
        for k,v in info.items():
            metadata[k] = v   
    
    if "bands" not in metadata:
        metadata["bands"] = ee_img.bandNames().getInfo()

    download_name = os.path.splitext(os.path.basename(local_dir))[0] + str(int(random.random() * 1000))

    json_file = os.path.join(local_dir, 'info.json')
    with open(json_file, 'w') as outfile:
        json.dump(metadata, outfile)

    download.MaybeDownloadWithTask(ee_img,
                                   download_name,region=region,
                                   path=local_dir)


    os.rename(os.path.join(local_dir ,download_name + ".tif"),
              os.path.join(local_dir , RASTER_NAME+ ".tif"))
    

    return
    
