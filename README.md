# `ee_ipl_uv` package

This project contains a python package on `ee_ipl_uv` folder.

The package extends the functionality of [Google Earth Engine API](https://developers.google.com/earth-engine/#api) (`ee`) to
implement cloud detection algorithms.

In particular it contains the code to reproduce the results of ([Mateo-Garcia et al 2018](http://dx.doi.org/10.3390/rs10071079)) and ([Gomez-Chova et al 2017](http://dx.doi.org/10.1117/1.JRS.11.015005)).


## Installation

The package is tested on an standard python3 anaconda environment. It requires to install first the `earthengine-api` package.

```bash
python setup.py install
```

## `reproducibility` folder

It contains the script, the notebooks and instructions needed to reproduce the results of ([Mateo-Garcia et al 2018](http://dx.doi.org/10.3390/rs10071079)). See [reproducibility/README.md](reproducibility/README.md)

## `examples` folder

The `examples` folder contains several notebooks that go step by step in the proposed multitemporal cloud detection schemes.
* The notebook `cloudscore_different_preds.ipynb` shows ready to use examples of the proposed cloud detection scheme. 
The function `CloudClusterScore` receives an image as an argument and return the proposed cloud score together with the 
background estimation. This notebook shows also the compairson between the proposed cloud masks and the ground truth for a given patch.
* The notebook `multitemporal_cloud_masking_sample.ipynb` explains in great detail the method for background estimation 
proposed in ([Gomez-Chova et al 2017](http://dx.doi.org/10.1117/1.JRS.11.015005))
* The notebook `clustering_differences.ipynb` explains the clustering procedure and the thresholding of the image to form the cloud mask.

Scheme of the proposed methodology:
![alt text](esquema_GEE.png)

If you use this code please cite:
 
 ```
@article{mateo-garcia_multitemporal_2018,
 author = {Mateo-García, Gonzalo and Gómez-Chova, Luis and Amorós-López, Julia and Muñoz-Marí, Jordi and Camps-Valls, Gustau},
 doi = {10.3390/rs10071079},
 journal = {Remote Sensing},
 language = {en},
 link = {http://www.mdpi.com/2072-4292/10/7/1079},
 month = {jul},
 number = {7},
 pages = {1079},
 title = {Multitemporal {Cloud} {Masking} in the {Google} {Earth} {Engine}},
 urldate = {2018-07-10},
 volume = {10},
 year = {2018}
} 
 ```

  
