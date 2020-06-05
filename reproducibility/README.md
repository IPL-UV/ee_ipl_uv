# Reproducibility

This folder contains the script, the notebooks and instructions needed to reproduce the results of [Mateo-Garcia et al 2018: Multitemporal Cloud Masking in the Google Earth Engine](http://dx.doi.org/10.3390/rs10071079).

## Load the same patches from Biome dataset

The 2661 patches used in the study are stored in file `splits_slices_biome.json` the following python snippet opens this `json` file and loads an image patch from the Biome dataset ([Foga et al 2017](http://doi.org/10.5066/F7251GDH)).

```python
import json
from skimage.external import tifffile
import matplotlib.pyplot as plt


with open("splits_slices_biome.json","r") as f:
    slices_patches = json.load(f)

def to_slice(slice_list):
    return tuple([slice(*s) for s in slice_list])

product_name = "LC80150312014226LGN00"
patch_name = "009_006"
biome_image_path = "/path/to/biome/dataset/BC/"+product_name+"/"+product_name+"_B1.TIF"
slice_to_read = to_slice(slices_patches[product_name][patch_name])

# load from the image product_name the 500x500 slice corresponding to patch 009_006 
img = tifffile.imread(biome_image_path,memmap=True)[slice_to_read]

plt.figure(figsize=(10,10))
plt.imshow(img)
_ = plt.title(product_name+" B1 500x500 patch: "+patch_name)
``` 

## Rerun the experiments

### Run and download the data

The script `reproducibility.py` reruns the experiments. It applies the proposed methodology and download for each patch the Landsat 8 product and the ground truth from the Biome dataset ([Foga et al 2017](http://doi.org/10.5066/F7251GDH)).

In case you don't need to download the data from the GEE see `examples/cloudscore_different_preds.ipynb`. 

To run `reproducibility.py` you need to install the packages `pydrive`. The `pydrive` package is used to download the data from Google Drive. It expects the folder `ee_ipl_uv_downloads` exists in your Google Drive. 

To donwnload a single patch run:

```bash
mkdir reproducibility_results
python reproducibility.py DownloadImageResults --image-index LC80290372013257LGN00 --split 013_011 --method percentile --basepath reproducibility_results
```

To download all the patches:

```bash
mkdir reproducibility_results
python reproducibility.py DownloadAll --method percentile --basepath reproducibility_results
```

### Analyisis of the downloaded data

See the notebooks `analyisis_reproducibility_example.ipynb` and `global_analyisis_reproducibility.ipynb`.