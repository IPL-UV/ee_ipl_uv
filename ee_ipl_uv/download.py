'''
Created on May 12, 2016

@author: Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es
'''

import os
import re
import zipfile
import numpy as np
import datetime
import ee
import requests
from collections import namedtuple
import time
import tempfile
from ee_ipl_uv.file_utils import addFormat, createTempFile
import shutil
import logging

logger = logging.getLogger(__name__)

def AuthDrive():
    import yaml
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    path_script = os.path.dirname(__file__)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml") as fout:
        DEFAULT_SETTINGS = {
            'client_config_backend': 'file',
            'client_config_file': os.path.join(path_script, 'client_secrets.json'),
            'save_credentials': False,
            'oauth_scope': ['https://www.googleapis.com/auth/drive']
        }
        fout.write(yaml.dump(DEFAULT_SETTINGS))
        fout.flush()
        gauth = GoogleAuth(fout.name)
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            # Athenticate if they're not there
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        # Save the current credentials to a file
        gauth.SaveCredentialsFile("mycreds.txt")

        drive = GoogleDrive(gauth)
    return drive


def MaybeExtract(image_name_dir, image_name_zip):
    """Extract de image image_name_zip if it has not been extracted"""
    if os.path.exists(image_name_dir):
        return image_name_dir
    os.makedirs(image_name_dir)
    zip_obj = zipfile.ZipFile(image_name_zip)
    zip_obj.extractall(image_name_dir)
    return image_name_dir


def MaybeDownloadThumb(image, params={"format": "jpg"}, image_name=None,
                       path=os.getcwd(),
                       force=False,
                       footprint=None):
    """
    Download thumb on path folder.

    :param image: image to download the thumb
    :param params: params to getThumbUrl()
    :param image_name: name of the image to download. Create temp file if None
    :param path:
    :param force: (optional) overwrite image_name if exists
    :param footprint: string to print into the image
    :return: Returns the downloaded file full path
    """

    params = dict(params)
    filecreated = False
    if image_name is None:
        image_name = createTempFile(params, "thumb_", path)
        filecreated = True

    if not force and not filecreated and os.path.exists(image_name):
        return image_name

    if not filecreated and os.path.exists(image_name):
        os.remove(image_name)

    url = image.getThumbURL(params)

    r_link = requests.get(url, stream=True)
    if r_link.status_code == 200:
        with open(image_name, 'wb') as f:
            r_link.raw.decode_content = True
            shutil.copyfileobj(r_link.raw, f)
    else:
        raise Exception("Can't download Status: %d\n %s"%(r_link.status_code,r_link.text))

    if os.stat(image_name).st_size < 100:
        file_content = open(image_name).readline()
        raise Exception("File downloaded '{}' is annomally small. first line: '{}'".format(image_name,file_content))

    if footprint is not None:
        from PIL import Image
        from PIL import ImageDraw
        img = Image.open(image_name)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), footprint, (255, 255, 255))
        img.save(image_name)

    return image_name


def MaybeDownload(image, image_name_dir=None, path=os.getcwd(), remove_zip=False):
    """Download the ee.Image object at path with name image_name_dir"""
    if image_name_dir is None:
        image_info = image.getInfo()
        pattern = re.compile("/")
        image_name_dir = path+"/"+pattern.sub("-", image_info["id"])
    
    image_name_zip = image_name_dir+".zip"
    if os.path.isfile(image_name_zip):
        MaybeExtract(image_name_dir, image_name_zip)
        if remove_zip:
            os.remove(image_name_zip)
        return image_name_dir
    
    url = image.getDownloadURL()
    # print("link: "+url)
    r_link = requests.get(url, stream=True)
    if r_link.status_code == 200:
        with open(image_name_zip, 'wb') as f:
            r_link.raw.decode_content = True
            shutil.copyfileobj(r_link.raw, f)
    else:
        raise Exception("Can't download Status: %d"%r_link.status_code)

    #image_name_zip, cabecera = urlretrieve(url, image_name_zip)
    
    statinfo = os.stat(image_name_zip)
    if statinfo.st_size < 100:
        file_content = open(image_name_zip).readline()
        raise Exception("File downloaded {} is annomally small. first line: '{}'".format(image_name_zip,file_content))

    logger.info("Downloaded {1}: {0:.2f}MB".format(statinfo.st_size/1e6, image_name_zip))
    MaybeExtract(image_name_dir, image_name_zip)
    if remove_zip:
        os.remove(image_name_zip)
    return image_name_dir


def MaybeDownloadWithTask(image, image_name,region=None, path=os.getcwd(), force=False):
    """ Download the image to google drive and then from Google Drive to path

    Note: image is downloaded as a geotif file.

    :param image: server image object to download
    :type image: ee.Image
    :param image_name: image file name if exists and not force it doesnt download anything
    :param path: path to download the image
    :param force: if we want to force the donwload (overwrites if exist)
    :return: the full path to the downloaded image
    """

    image_name_full = os.path.join(path,addFormat(image_name, "tif"))
    if os.path.isfile(image_name_full) and not force:
        return image_name_full

    task = ee.batch.Export.image.toDrive(image,
                                         region=region,
                                         description=image_name,
                                         folder="ee_ipl_uv_downloads")
    task.start()
    return WaitAndDownload(task, image_name, "tif", path, True)


def WaitTask(task, extra_text=""):
    """
    Wait until task finishes

    :param task:
    :param extra_text: Extra text to add to log
    :return:
    """
    time_elapsed = 0
    while task.active():
        if (time_elapsed % 60) == 0:
            logger.info("{} Elapsed: {:d}s Status: {}".format(extra_text, time_elapsed,
                                                              task.status()["state"]))
        time.sleep(10)
        time_elapsed += 10

    if task.status()["state"] != "COMPLETED":
        raise IOError("{} Task status is not COMPLETED: {}".format(extra_text, repr(task.status())))

    logger.info("{} ee.task COMPLETED {}s".format(extra_text, time_elapsed))


def WaitAndDownload(task, filename, formato="tif",
                    path=os.getcwd(), force=False):
    """
    Wait until task finishes and download the file from Drive afterwards

    :param task:
    :param filename:
    :param formato:
    :param path:
    :param force:
    :return: name of the downloaded item
    """
    WaitTask(task)
    return DownloadFromDrive(filename, formato, path, force)


def DownloadFromDrive(file_name, formato="tif",  path=os.getcwd(), force=False):
    """ Downloads an image from google drive and deletes it at the end

    :param file_name: file to download
    :param formato: formato de el asset a descargar
    :param path: path to download the image
    :param force: if we want to force the donwload (overwrites if exist)
    :return: the full path to the donwloaded image
    """
    image_name_full_original = os.path.join(path, addFormat(file_name, formato))
    if os.path.isfile(image_name_full_original) and not force:
        return image_name_full_original
    # Download file from drive
    drive = AuthDrive()
    file_list = drive.ListFile({'q': "title contains '{}' and trashed=false"
                               .format(file_name)}).GetList()

    #expr = re.compile("%s-(\d+)-(\d+)\.%s" % (file_name, formato))
    expr = re.compile("(%s)(-\d+-\d+)?\.%s" % (file_name, formato))

    f_downs = []
    for file_down in file_list:
        title = file_down.attr["metadata"]["title"]
        fmo = expr.fullmatch(title)
        if fmo is not None:
            image_name_full = os.path.join(path, addFormat(title, formato))

            logger.info("Downloading image %s from drive"%title)
            file_down.GetContentFile(image_name_full)

            f_downs.append(image_name_full)

            # Delete image from drive
            filele = drive.CreateFile()
            filele.auth.service.files().delete(fileId=file_down['id']).execute()

    if len(f_downs) == 1:
        return f_downs[0]

    if len(f_downs) == 4:
        ## Case when the image is divided in 4 patches.
        from skimage.external import tifffile
        f_downs = sorted(f_downs)
        up = np.concatenate([tifffile.imread(f) for f in f_downs[:2]],
                            axis=1)
        down = np.concatenate([tifffile.imread(f) for f in f_downs[2:]],
                            axis=1)

        tifffile.imsave(image_name_full_original,
                        np.concatenate((up,down),axis=0))

        for fd in f_downs:
            os.remove(fd)

        return image_name_full_original

    raise IOError("files {} dont know how to concat them".format(f_downs))


def DownloadImageCollectionThumb(img_collection, params={"format": "jpg"},
                                 image_name_prepend="imagen_thumb",
                                 path=os.getcwd(),
                                 add_timestamp=True):
    """Download the imageCollection images as thumbs images. Returns the path of them together with the time"""
    size = img_collection.size().getInfo()
    logger.info("Downloading: {} images:".format(size))
    lista = img_collection.toList(size)
    if "format" not in params:
        params["format"] = "jpg"
    image_name_dir = []
    time_start = []
    for i in range(0, size):
        imagen = ee.Image(lista.get(i))
        image_file = tempfile.NamedTemporaryFile(suffix="."+params["format"],
                                                 prefix=image_name_prepend + "_" + str(i)+"_",
                                                 delete=False,
                                                 dir=path)
        image_file.close()
        timestamp_it = datetime.datetime.utcfromtimestamp(imagen.get("system:time_start").getInfo() / 1000) \
            .strftime("%Y-%m-%d %H:%M:%S")
        footprint = timestamp_it if add_timestamp else None
        image_name_dir.append(MaybeDownloadThumb(imagen, params, image_file.name, force=True,footprint=footprint))
        time_start.append(timestamp_it)

    DownloadedThumbImageCollection = namedtuple("DownloadedThumbImageCollection", ["image_name", "time_start"])
    return DownloadedThumbImageCollection(image_name_dir, time_start)


def DownloadImageCollection(img_collection):
    """Download a complete ee.ImageCollection()"""
    size = img_collection.size().getInfo()
    logger.info("Downloading: {} images:".format(size))
    lista = img_collection.toList(size)
    image_name_dir = []
    time_start = []
    for i in range(0, size):
        imagen = ee.Image(lista.get(i))
        image_name_dir.append(MaybeDownload(imagen))
        timeseries = datetime.datetime.utcfromtimestamp(imagen.get("system:time_start").getInfo()/1000)
        time_start.append(timeseries)
    
    DownloadedImageCollection = namedtuple("DownloadedImageCollection", ["image_name", "time_start"])
    return DownloadedImageCollection(image_name_dir, time_start)


def MosaicImageList(list_images, dims, image_name=None):
    from PIL import Image
    """It creates a dims[0] x dims[1] mosaic with the images on list_images"""
    images = [Image.open(image) for image in list_images]
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', ((max_width+1)*dims[1], (max_height+1)*dims[0]))

    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            indice = i*dims[1]+j
            if indice < len(images):
                im = images[indice]
                new_im.paste(im, (max_width*j, max_height*i))
                im.close()

    if image_name is None:
        image_name = createTempFile(params={"format": "jpg"}, prefix="mosaic_")

    new_im.save(image_name)
    new_im.close()
    return image_name

