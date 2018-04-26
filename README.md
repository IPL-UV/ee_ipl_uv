# `ee_ipl_uv` package

This project contains a python package on `ee_ipl_uv` folder.

The package extends the functionality of [Google Earth Engine API](https://developers.google.com/earth-engine/#api) (`ee`) to
implement cloud detection algorithms.

In particular it contains the code to reproduce the results of [Mateo-Garcia et al. (submitted)]() and [Gomez-Chova et al 2017](http://dx.doi.org/10.1117/1.JRS.11.015005).


## Installation
The package is tested on an standard python3 anaconda environment. It requires to install first the `earthengine-api` package.

```bash
python setup.py install
```
## `examples` folder
The `examples` folder contains several notebooks that go step by step in the proposed multitemporal cloud detection schemes.
* The notebook `cloudscore_different_preds.ipynb` shows ready to use examples of the proposed cloud detection scheme. 
The function `CloudClusterScore` receives an image as an argument and return the proposed cloud score together with the 
background estimation. 
* The notebook `multitemporal_cloud_masking_sample.ipynb` explains in great detail the method for background estimation 
proposed in [Gomez-Chova et al 2017](http://dx.doi.org/10.1117/1.JRS.11.015005)
* The notebook `clustering_differences.ipynb` explains the clustering procedure and the 
thresholding of the image to form the cloud mask.

If you use this code please cite:
 
 ```
 Bibtex: @article { GChova2017,
  author = { Gómez-Chova, Luis and Amorós-López, Julia and Mateo-García, Gonzalo and Muñoz-Marí, Jordi and Camps-Valls, Gustau } ,
  title = { Cloud masking and removal in remote sensing image time series } ,
  journal = { Journal of Applied Remote Sensing } ,
  volume = { 11 } ,
  number = { 1 } ,
  pages = { 015005 } ,
  year = { 2017 } ,
  isbn = { 1931-3195 } ,
  doi = { http://dx.doi.org/10.1117/1.JRS.11.015005 } 
   } 
 ```

  
