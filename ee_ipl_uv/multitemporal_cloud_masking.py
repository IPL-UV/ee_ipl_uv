'''
Created on June 12, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''

import ee
import random
from datetime import datetime
from datetime import timedelta
from ee_ipl_uv import kernel
from ee_ipl_uv import converters
from ee_ipl_uv import time_series_operations
from ee_ipl_uv import normalization
from ee_ipl_uv import clustering
from ee_ipl_uv import predefined_cloud_algorithms
import numpy as np
import logging

CC_IMAGE_TOP = .6

logger = logging.getLogger(__name__)

class ModelCloudMasking:
    def __init__(self, img, bands, cloud_mask, max_lags,
                 region, seed=None, beta=None):
        """
        Class which implements linear and kernel models on Google Earth Engine
        for image forecasting as described in XXX paper

        :param img: to apply the model
        :type img: ee.Image
        :param bands: list with the names of the bands
        :type bands: list[str]
        :param cloud_mask: cloud_mask image to mask pixels for "estimation"
        :type cloud_mask: ee.Image
        :param max_lags: number of lags of the model
        :param beta: beta of the model (ponderate estimation vs prediction)
        :param region: geom to fit the model (model will be fitted only with pixels of this geom)

        :type region: ee.Geometry
        """
        self.img = img
        self.bands = bands
        self.cloud_mask = cloud_mask
        self.max_lags = max_lags
        self.region = region
        bands_modeling_estimation = list(self.bands)
        for lag in range(1, self.max_lags):
            bands_lag = list(map(lambda x: x + "_lag_" + str(lag), self.bands))
            bands_modeling_estimation.extend(bands_lag)
        self.bands_modeling_estimation = bands_modeling_estimation

        bands_modeling_prediction = []
        for lag in range(1, self.max_lags + 1):
            bands_lag = list(map(lambda x: x + "_lag_" + str(lag), self.bands))
            bands_modeling_prediction.extend(bands_lag)

        self.bands_modeling_prediction = bands_modeling_prediction
        self.bands_modeling_estimation_input = list(filter(lambda band: band.find("lag") != -1,
                                                      self.bands_modeling_estimation))
        self.bands_modeling_estimation_output = list(filter(lambda band: band.find("lag") == -1,
                                                       self.bands_modeling_estimation))

        self.bands_modeling_prediction_input = list(filter(lambda band: band.find("lag_1") == -1,
                                                      self.bands_modeling_prediction))
        self.bands_modeling_prediction_output = list(filter(lambda band: band.find("lag_1") != -1,
                                                       self.bands_modeling_prediction))
        if seed is None:
            self.seed = random.randint(0, 36000)
        else:
            self.seed = seed

        # Calculamos CC de la imagen actual
        dictio = self.cloud_mask.select([0], ["cloud"]).reduceRegion(reducer=ee.Reducer.mean(), geometry=region,
                                                                     bestEffort=True)
        self.cc_image = ee.Number(dictio.get("cloud")).getInfo()

        self.img_est = self.img.updateMask(self.cloud_mask.eq(0))
        self.img_est = self.img_est.select(self.bands_modeling_estimation)

        self.img_pred = self.img.select(self.bands_modeling_prediction)

        # Decrease beta linearly as the amount of cc_image increases
        if beta is None:
            self.beta = .5 - .5*self.cc_image/CC_IMAGE_TOP
            self.beta = self.beta if self.beta > 0 else 0
        else:
            self.beta = beta

        self.alpha = None
        self.kernel_rbf = None
        self.gamma = None
        self.lmbda = None

        self.omega = None
        self.intercept = None

    def _BuildDataSet(self, sampling_factor,
                      normalize):

        estimation_set = self.img_est.sample(region=self.region,
                                             factor=sampling_factor,
                                             seed=self.seed)

        prediction_set = self.img_pred.sample(region=self.region,
                                              factor=sampling_factor,
                                              seed=self.seed)
        # Add weights
        estimation_set_size = ee.Number(estimation_set.size())
        prediction_set_size = ee.Number(prediction_set.size())
        peso_estimacion = ee.Number(self.beta).divide(estimation_set_size)
        peso_prediccion = ee.Number(1-self.beta).divide(prediction_set_size)

        bands_modeling_estimation_input_weight = list(self.bands_modeling_estimation_input)
        bmp = list(self.bands_modeling_prediction)
        bmp.append("weight")
        bme = list(self.bands_modeling_estimation)
        bme.append("weight")

        estimation_set = estimation_set.map(
            lambda ft: ee.Feature(ft).set("weight", peso_estimacion))
        prediction_set = prediction_set.map(
            lambda ft: ee.Feature(ft).set("weight", peso_prediccion))
        bands_modeling_estimation_input_weight.append("weight")

        self.estimation_set = estimation_set
        self.prediction_set = prediction_set
        if (self.beta > 0) and (self.cc_image < CC_IMAGE_TOP):
            self.datos = estimation_set.merge(prediction_set.select(bmp,
                                                                    bme))
        else:
            logger.info("Using only prediction")
            self.datos = prediction_set.select(bmp,
                                               bme)
        if normalize:
            self.datos, self.inputs_mean, self.inputs_sd = normalization.ComputeNormalizationFeatureCollection(
                self.datos, self.bands_modeling_estimation_input, only_center_data=False, weight="weight")
            self.datos, self.outputs_mean, self.outputs_sd = normalization.ComputeNormalizationFeatureCollection(
                self.datos, self.bands_modeling_estimation_output, only_center_data=True, weight="weight")

        self.inputs = self.datos.select(bands_modeling_estimation_input_weight)
        self.outputs = self.datos.select(self.bands_modeling_estimation_output)

        return

    def TrainRBFKernelLocal(self, lmbda=.1, gamma=.5, sampling_factor=1./100.,
                            with_task=False, with_cross_validation=False):
        """
        Fit Kernelized RBF Ridge Regression to the current image subsampled. It downloads the data to
        fit. It uses sklearn.kernel_ridge library.

        :param lmbda: regularization factor
        :param gamma: gamma factor to build the kernel (https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
        :param sampling_factor: percentage of pixels to sample to build the model
        :param with_cross_validation: donwload the data to fit the model with a task
        :param with_task: donwload the data to fit the model with a task

        :return:
        """
        self._BuildDataSet(sampling_factor, normalize=True)
        bme = list(self.bands_modeling_estimation)
        bme.append("weight")

        ds_download_pd = converters.eeFeatureCollectionToPandas(self.datos, bme,
                                                                with_task=with_task)
        logger.info("Size of downloaded ds: ", ds_download_pd.shape)

        pesos = np.asanyarray(ds_download_pd.weight)
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        
        n_samples = ds_download_pd.shape[0]
        ini = datetime.now()                

        if with_cross_validation:
            kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                              param_grid={"alpha": np.logspace(-5, 2, 8),
                                          "gamma": np.logspace(-3, 2, 6)},
                              fit_params={"sample_weight": pesos})
            kr.fit(ds_download_pd[self.bands_modeling_estimation_input],
                   ds_download_pd[self.bands_modeling_estimation_output])
            logger.info("CV params:", kr.best_params_)
            self.alpha = ee.Array(kr.best_estimator_.dual_coef_.tolist())  # column vector
            self.kernel_rbf = kernel.Kernel(self.inputs, self.bands_modeling_estimation_input,
                                            kernel.RBFDistance(kr.best_params_["gamma"]),
                                            weight_property="weight")
            self.gamma = kr.best_params_["gamma"]
            self.lmbda = kr.best_params_["alpha"]

        else:
            ridge_kernel = KernelRidge(alpha=lmbda,
                                       gamma=gamma,
                                       kernel="rbf")
            self.gamma = gamma
            self.lmbda = lmbda

            ridge_kernel.fit(ds_download_pd[self.bands_modeling_estimation_input],
                             ds_download_pd[self.bands_modeling_estimation_output],
                             sample_weight=pesos)
            self.alpha = ee.Array(ridge_kernel.dual_coef_.tolist())  # column vector
            self.kernel_rbf = kernel.Kernel(self.inputs, self.bands_modeling_estimation_input,
                                            kernel.RBFDistance(gamma),
                                            weight_property="weight")
        duration = (datetime.now() - ini).total_seconds()
        
        return n_samples,duration

    def TrainRBFKernelServer(self, lmbda=.1, gamma=50., sampling_factor=1./1000.):
        """
        Fit Kernelized RBF Ridge Regression to the current image on the server
        :param lmbda:
        :param gamma: gamma factor to build the kernel (https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
        :param sampling_factor: percentage of pixels to sample to build the model
        :return:
        """
        self._BuildDataSet(sampling_factor, normalize=True)
        inputs = self.inputs
        outputs = self.outputs
        labels_input = self.bands_modeling_estimation_input
        labels_output = self.bands_modeling_estimation_output

        # Train
        self.kernel_rbf = kernel.Kernel(inputs, labels_input,
                                        kernel.RBFDistance(gamma),
                                        weight_property="weight")

        outputs_eeArray = converters.eeFeatureCollectionToeeArray(outputs, labels_output)

        self.alpha = self.kernel_rbf.getAlphaeeArray(outputs_eeArray, lmbda)

    def PredictRBFKernel(self):
        """
        Fit  KRR to the (subsampled) image and apply the model to the full image
         to obtain the forecasted image

        :return: forecasted image
        :rtype ee.Image
        """
        labels_output = self.bands_modeling_estimation_output
        labels_input = self.bands_modeling_estimation_input

        # Test
        image_test_input = self.img.select(labels_input)
        image_test_input = normalization.ApplyNormalizationImage(image_test_input, labels_input,
                                                                 self.inputs_mean, self.inputs_sd)

        # image_test_output = self.img.select(labels_output)
        image_forecast_complete = self.kernel_rbf.applyModelToImage(image_test_input, self.alpha)

        # Convert arrayImage to Image
        image_forecast_complete = image_forecast_complete.arrayProject([1]).arrayFlatten([labels_output])

        # Denormalize outputs
        return self._DenormalizeAndRename(image_forecast_complete, True)

    def _DenormalizeAndRename(self, image_forecast_complete, normalize):
        if normalize:
            image_forecast_complete = normalization.ApplyDenormalizationImage(image_forecast_complete,
                                                                              self.bands_modeling_estimation_output,
                                                                              self.outputs_mean)
        # Rename labels
        forecast_labels = list(map(lambda prop: prop + "_forecast", self.bands_modeling_estimation_output))
        image_forecast_complete = image_forecast_complete.select(self.bands_modeling_estimation_output,
                                                                 forecast_labels)

        return image_forecast_complete

    def TrainLinearLocal(self, lmbda=.1, sampling_factor=1./100.,
                         with_task=False, normalize=True, divide_lmbda_by_n_samples=False,
                         with_cross_validation=False):
        """
        Fit linear ridge regression to the image downloading the data and
        doing the fitting with sklearn.linear_model.

        :param lmbda:
        :param sampling_factor:
        :param divide_lmbda_by_n_samples:
        :param normalize: if we want normalization
        :param with_task: if the download is done using a task (require pydrive)
        :param with_cross_validation: donwload the data to fit the model with a task
        :return: forecasted image
        :rtype ee.Image
        """
        self._BuildDataSet(sampling_factor, normalize=False)
        bme = list(self.bands_modeling_estimation)
        bme.append("weight")

        ds_download_pd = converters.eeFeatureCollectionToPandas(self.datos, bme,
                                                                with_task=with_task)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV

        if divide_lmbda_by_n_samples:
            lmbda = lmbda/ds_download_pd.shape[0]
        
        n_samples = ds_download_pd.shape[0]
        
        logger.info("Size of downloaded ds: ", ds_download_pd.shape)
        pesos = np.asanyarray(ds_download_pd.weight)
        ini = datetime.now()
        if with_cross_validation:
            kr = GridSearchCV(Ridge(fit_intercept=True,
                                    alpha=lmbda,
                                    normalize=normalize), cv=5,
                              param_grid={"alpha": np.logspace(-6, 2, 9)},
                              fit_params={"sample_weight": pesos})
            kr.fit(ds_download_pd[self.bands_modeling_estimation_input],
                   ds_download_pd[self.bands_modeling_estimation_output])

            lin_reg = kr.best_estimator_
            self.lmbda = kr.best_params_["alpha"]
            logger.info("CV params:", kr.best_params_)
        else:
            self.lmbda = lmbda
            lin_reg = Ridge(fit_intercept=True,
                            alpha=lmbda,
                            normalize=normalize)
            lin_reg.fit(ds_download_pd[self.bands_modeling_estimation_input],
                        ds_download_pd[self.bands_modeling_estimation_output],
                        pesos)

        self.omega = ee.Array(lin_reg.coef_.T.tolist())
        self.intercept = ee.Array([lin_reg.intercept_.tolist()])
        
        duration = (datetime.now() - ini).total_seconds()

        return n_samples,duration

    def TrainLinearServer(self, lmbda=.1, sampling_factor=1./100.):
        """
        Fit linear ridge regression to the image

        :param lmbda:
        :param sampling_factor:
        :return: forecasted image
        :rtype ee.Image
        """
        self._BuildDataSet(sampling_factor, normalize=True)
        inputs = self.inputs
        outputs = self.outputs
        labels_input = self.bands_modeling_estimation_input
        labels_output = self.bands_modeling_estimation_output

        # Train
        outputs_eeArray = converters.eeFeatureCollectionToeeArray(outputs, labels_output)
        inputs_eeArray = converters.eeFeatureCollectionToeeArray(inputs, labels_input)
        weight_eeArray = converters.eeFeatureCollectionToeeArray(inputs, ["weight"])
        weight_matrix = weight_eeArray.matrixToDiag()
        inputs_weigted = inputs_eeArray.matrixTranspose().matrixMultiply(weight_matrix)

        tikhonov = ee.Array.identity(len(labels_input)).multiply(lmbda).multiply(ee.Number(inputs.size()))

        omega = inputs_weigted.matrixMultiply(inputs_eeArray).add(tikhonov)\
            .matrixSolve(inputs_weigted.matrixMultiply(outputs_eeArray))

        array_sd = ee.Array(self.inputs_sd.values(labels_input)).pow(-1)
        array_sd = ee.Array.cat([array_sd], 1).matrixToDiag()

        self.omega = array_sd.matrixMultiply(omega)

        prediction_mean = ee.Array([ee.Dictionary(self.inputs_mean).values(labels_input)]).matrixMultiply(self.omega)

        self.intercept = ee.Array([ee.Dictionary(self.outputs_mean).values(labels_output)]).subtract(prediction_mean)

        return

    def PredictLinear(self):
        """
        Apply the model to the current image.
        Must call before TrainLinearLocal or TrainLinearServer

        :return:
        """
        assert self.omega is not None, "Model has not been trained"
        labels_input = self.bands_modeling_estimation_input
        labels_output = self.bands_modeling_estimation_output
        # Test
        image_test_input = self.img.select(labels_input)

        array1D = image_test_input.toArray()
        array2D = array1D.toArray(1)

        # 1 x num_bands_output
        image_forecast_complete = array2D.matrixTranspose().matrixMultiply(self.omega).add(self.intercept)

        # Convert arrayImage to Image
        image_forecast_complete = image_forecast_complete.arrayProject([1]).arrayFlatten([labels_output])

        # Denormalize outputs
        # Denormalize outputs
        return self._DenormalizeAndRename(image_forecast_complete, False)


def PreviousImagesWithCC(landsat_img, region_of_interest=None,
                         REVISIT_DAY_PERIOD=15, NUMBER_IMAGES=30,
                         include_img=False):
    """
    Return the NUMBER_IMAGES previous images with cloud cover

    :param landsat_img:
    :param region_of_interest:
    :param REVISIT_DAY_PERIOD:
    :param NUMBER_IMAGES:
    :param include_img: if the current image (landsat_img) should be included in the series
    :return:
    """
    # Get collection id
    landsat_info = landsat_img.getInfo()
    landsat_full_id = landsat_info['id']
    landsat_image_index = landsat_info['properties']['system:index']
    landsat_collection = landsat_full_id.replace("/"+landsat_image_index, "")

    WRS_ROW = landsat_info['properties']['WRS_ROW']
    WRS_PATH = landsat_info['properties']['WRS_PATH']

    current_date = datetime.utcfromtimestamp(landsat_info['properties']['system:time_start']/1000)
    searching_date = current_date-timedelta(days=REVISIT_DAY_PERIOD*NUMBER_IMAGES)

    # Retrieve previous images
    if region_of_interest is None:
        region_of_interest = ee.Element.geometry(landsat_img)
        landsat_collection = ee.ImageCollection(landsat_collection)\
            .filter(ee.Filter.eq("WRS_ROW", WRS_ROW)) \
            .filter(ee.Filter.eq("WRS_PATH", WRS_PATH))
    else:
        landsat_collection = ee.ImageCollection(landsat_collection)\
            .filterBounds(region_of_interest)
            #.filter(ee.Filter.contains(leftField='.geo',
            #                           rightValue=))

    imgColl = landsat_collection.filterDate(ee.Date(searching_date),
                                            ee.Date(current_date+timedelta(seconds=10)))\
        .sort("system:time_start")

    if not include_img:
        imgColl = imgColl.filter(ee.Filter.neq('system:index', landsat_image_index))

    def ComputeCloudCoverGeom(img):
        """Compute mean cloud cover and add it as property to the image"""
        clouds = ee.Algorithms.Landsat.simpleCloudScore(img).gte(50)
        # clouds_original = image_predict_clouds.select("fmask").eq(2)
        # clouds_original = clouds_original.where(image_predict_clouds.select("fmask").eq(4),2)

        # Add growing to the mask

        # clouds = clouds_original.reduceNeighborhood(ee.Reducer.max(),
        #                                            ee.Kernel.circle(radius=3))
        # clouds = ee.Algorithms.Landsat.simpleCloudScore(img).select("cloud").gt(50).toFloat()

        dictio = clouds.reduceRegion(reducer=ee.Reducer.mean(), geometry=region_of_interest,
                                     bestEffort=True)
        numerito = ee.Number(dictio.get("cloud")).multiply(100)
        img = img.set("CC", numerito)
        return img

    # Filter images with CC lower than THRESHOLD
    return imgColl.map(ComputeCloudCoverGeom)

LANDSAT8_BANDNAMES=['B1',  'B2','B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'BQA']

def SelectImagesTraining(landsat_img,band_names=LANDSAT8_BANDNAMES,
                         region_of_interest=None, num_images=3, REVISIT_DAY_PERIOD=15, threshold_cc=5):
    """
    Given a landsat image, it returns the num_images previous images together with the current image with CC lower than
     THRESHOLD_CC. The returned image contains previous images in bands. It will have num_bands*(num_images+1) bands

    :param landsat_img: landsat image
    :type landsat_img: ee.Image
    :param band_names: band names to add to the current image
    :param region_of_interest:
    :type region_of_interest: ee.Geometry
    :param num_images:
    :param REVISIT_DAY_PERIOD:
    :param threshold_cc:
    :return:
    :rtype ee.Image
    """
    imgColl = PreviousImagesWithCC(landsat_img,region_of_interest,REVISIT_DAY_PERIOD)
    imgColl = imgColl.filter(ee.Filter.lt("CC", threshold_cc))
    size_img_coll = ee.Number(imgColl.size())
    imagenes_training = imgColl.toList(count=num_images, offset=size_img_coll.subtract(num_images))

    # join images into a single image
    # band_names = landsat_img.bandNames()
    for lag in range(1, num_images+1):
        image_add = ee.Image(imagenes_training.get(num_images-lag))
        new_band_names = time_series_operations.GenerateBandNames(band_names, "_lag_"+str(lag))
        image_add = image_add.select(band_names, new_band_names)
        landsat_img = landsat_img.addBands(image_add)
        landsat_img = landsat_img.set("system:time_start_lag_"+str(lag), image_add.get("system:time_start"))
        landsat_img = landsat_img.set("CC_lag_" + str(lag), image_add.get("CC"))

    return landsat_img

def PredictPercentile(img, region_of_interest,num_images=3,threshold_cc=5):
    imgColl = PreviousImagesWithCC(img, region_of_interest)
    imgColl = imgColl.filter(ee.Filter.lt("CC", threshold_cc)).limit(num_images)

    img_percentile = imgColl.reduce(reducer=ee.Reducer.percentile(percentiles=[50]))
    return img_percentile

PARAMS_CLOUDCLUSTERSCORE_DEFAULT = {"threshold_cc":5,
                                    "sampling_factor":.05,
                                    "lmbda": 1e-6,
                                    "gamma": 0.01,
                                    "trainlocal": True,
                                    "with_task": True,
                                    "with_cross_validation": False,
                                    "threshold_dif_cloud":.09,
                                    "threshold_dif_shadow": .03,
                                    "numPixels":1000,
                                    "n_clusters":10
                                    }

def CloudClusterScore(img,region_of_interest,num_images=1,method_pred="persistence",
                      params=None):
    """
    Function to obtain the cloud cluster score in one shot.

    :param img:
    :param region_of_interest:
    :param num_images:
    :param method_pred:
    :param params:
    :return:  cloud mask (2: cloud,1: shadow, 0: clear)
    """
    if params is None:
        params = dict(PARAMS_CLOUDCLUSTERSCORE_DEFAULT)
    else:
        temp = dict(params)
        params = dict(PARAMS_CLOUDCLUSTERSCORE_DEFAULT)
        params.update(temp)

    image_with_lags = SelectImagesTraining(img,
                                           threshold_cc=params["threshold_cc"],
                                           region_of_interest=region_of_interest,
                                           num_images=num_images)

    clouds_original = predefined_cloud_algorithms.QACloudMask(img, strict=False)

    clouds = clouds_original.reduceNeighborhood(ee.Reducer.max(),
                                                ee.Kernel.circle(radius=3))

    reflectance_bands_landsat8 = ["B%d" % i for i in range(1, 12)]
    forecast_bands_landsat8 = [i + "_forecast" for i in reflectance_bands_landsat8]
    if method_pred == "persistence":
        reflectance_bands_landsat8_lag_1 = [i + "_lag_1" for i in reflectance_bands_landsat8]
        img_forecast = image_with_lags.select(reflectance_bands_landsat8_lag_1,forecast_bands_landsat8)

    elif method_pred == "percentile":
        img_percentile = PredictPercentile(img, region_of_interest,num_images=num_images)
        reflectance_bands_landsat8_perc50 = [i + "_p50" for i in reflectance_bands_landsat8]
        img_forecast = img_percentile.select(reflectance_bands_landsat8_perc50,
                                             forecast_bands_landsat8)

    elif method_pred == "linear":
        modelo = ModelCloudMasking(image_with_lags, reflectance_bands_landsat8,
                                   clouds, num_images, region_of_interest)
        if params["trainlocal"]:
            modelo.TrainLinearLocal(sampling_factor=params["sampling_factor"],
                                    lmbda=params["lmbda"],with_task=params["with_task"])
        else:
            modelo.TrainLinearServer(sampling_factor=params["sampling_factor"],
                                     lmbda=params["lmbda"])

        img_forecast = modelo.PredictLinear()

    elif method_pred == "kernel":
        modelo = ModelCloudMasking(image_with_lags, reflectance_bands_landsat8,
                                   clouds, num_images, region_of_interest)
        if params["trainlocal"]:
            modelo.TrainRBFKernelLocal(sampling_factor=params["sampling_factor"],
                                        lmbda=params["lmbda"], gamma=params["gamma"],
                                       with_cross_validation=params["with_cross_validation"])
        else:
            modelo.TrainRBFKernelServer(sampling_factor=params["sampling_factor"],
                                        lmbda=params["lmbda"], gamma=params["gamma"])

        img_forecast = modelo.PredictRBFKernel()

    else:
        raise NotImplementedError("method %s is not implemented"%method_pred)

    img_differences = image_with_lags.select(reflectance_bands_landsat8).subtract(
        img_forecast.select(forecast_bands_landsat8))

    clusterscore = clustering.ClusterClouds(img_differences,region_of_interest=region_of_interest,
                                            threshold_dif_cloud = params["threshold_dif_cloud"],
                                            threshold_dif_shadow = params["threshold_dif_shadow"],
                                            numPixels = params["numPixels"],
                                            n_clusters = params["n_clusters"])

    return clusterscore,img_forecast



FEATURES_CLOUDS = ["invNDVI",
                   "NDSI",
                   "Whiteness",
                   "Brightness_prob",
                   "HOT",
                   "swirnir",
                   "Vari_prob"]


def ComputeFeatures(img, sufix=""):
    """
    Compute features for cloud estimation:
    (invNDVI, NDSI, Whiteness, Brigthness prob, HOT, swirnir, Vari_prob)

    :param img:
    :type img: ee.Image
    :param sufix: sufix to access to bands: (B5+sufix). Generated bands will
    also have this sufix
    :return: image with layers as features. Name of those layers
    :rtype ee.Image, List[str]
    """
    band_names = list(map(lambda x: x+sufix, FEATURES_CLOUDS))
    # veg index (nir - red)/(nir+red)
    NDVI = img.normalizedDifference(['B5' + sufix, 'B4' + sufix])
    invNDVI = NDVI.subtract(1).abs()\
        .select([0], [FEATURES_CLOUDS[0]])

    # Snow index (green - swir)/ (green + swir)
    NDSI = img.normalizedDifference(['B3' + sufix, 'B6' + sufix])\
        .select([0], [FEATURES_CLOUDS[1]]).double()
    # Cast to double (otherwise float type and fails to download)

    # Whiteness
    mean = img.select(['B2' + sufix, 'B3' + sufix, 'B4' + sufix]).reduce(ee.Reducer.mean())
    whiteness = img.select(['B2' + sufix, 'B3' + sufix, 'B4' + sufix]).subtract(mean).abs()\
        .reduce(ee.Reducer.sum()).divide(mean)\
        .select([0], [FEATURES_CLOUDS[2]])

    # Brigthness prob
    # min(I(:,:, 5), 0.11) / 0.11
    # min(swir, 0.11) / 0.11
    Brightness_prob = img.select("B6" + sufix).min(.11).divide(.11)\
        .select([0], [FEATURES_CLOUDS[3]])

    # HOT = I(:,:, 1)-0.5 * I(:,:, 3)-0.08;
    # HOT= (blue - 0.5*red - 0.08)
    HOT = img.select("B2" + sufix).add(img.select("B4" + sufix).multiply(-.5)) \
        .add(-.08).select([0], [FEATURES_CLOUDS[4]])

    # B4_5 = nir ./((swir+0.01));
    swirnir = img.select("B5" + sufix).divide(img.select("B6" + sufix).add(.001))\
        .select([0], [FEATURES_CLOUDS[5]])

    # NDSIm(NDSI>0)=0;
    # NDVIm(NDVI<0)=0;
    NDSIm = NDSI.min(0)
    NDVIm = NDVI.max(0)

    # Vari_prob=1-max(abs(NDSIm),abs(NDVIm));
    Vari_prob = NDSIm.abs().max(NDVIm).multiply(-1).add(1)\
        .select([0], [FEATURES_CLOUDS[6]])

    return invNDVI.addBands(NDSI)\
        .addBands(whiteness).addBands(Brightness_prob)\
        .addBands(HOT).addBands(swirnir)\
        .addBands(Vari_prob), band_names


