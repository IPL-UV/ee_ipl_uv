'''
Created on June 15, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
import tempfile
import os


def addFormat(image_name, formato):
    if image_name.endswith(formato):
        return image_name
    return image_name+"."+formato


def removeFormat(image_name, formato):
    return image_name.replace("."+formato, "")


def createTempFile(params={"format": "jpg"}, prefix="thumb_", path=os.getcwd()):
    if "format" not in params:
        params["format"] = "jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False,
                                      suffix="." + params["format"], prefix=prefix,
                                      dir=path)
    tmp.close()
    return tmp.name