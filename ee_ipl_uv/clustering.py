from ee_ipl_uv import normalization
import ee

BANDS_MODEL = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]

def SelectClusters(img_differences,result_clustering,n_clusters, region_of_interest=None):
    """
    Function that contains the logic to create the cluster score mask. given the clustering result.

    :param img_differences:
    :param result_clustering:
    :param n_clusters:
    :param region_of_interest:
    :return:
    """
    bands_norm = ["B2", "B3", "B4"]

    # elms = dict()
    cloud_score = None

    for i in range(n_clusters):
        img_diff_clus = img_differences.updateMask(result_clustering.eq(i)).select(bands_norm)
        clusteri = img_diff_clus.reduceRegion(ee.Reducer.mean(),
                                              geometry=region_of_interest,
                                              bestEffort=True,
                                              scale=30)
        clusteri = clusteri.toArray(bands_norm)
        clusteri_mean = clusteri.reduce(ee.Reducer.mean(), axes=[0]).get([0])
        clusteri_norm = clusteri.multiply(clusteri).reduce(ee.Reducer.mean(),
                                                           axes=[0]).sqrt().get([0])

        val_clusteri = ee.Algorithms.If(clusteri_mean.gt(0), clusteri_norm, clusteri_norm.multiply(-1))
        cloud_scorei = result_clustering.eq(i).toFloat().multiply(ee.Number(val_clusteri))
        if cloud_score is None:
            cloud_score = cloud_scorei
        else:
            cloud_score = cloud_score.add(cloud_scorei)

    return cloud_score


def ClusterClouds(img_differences,threshold_dif_cloud=.045,
                  threshold_dif_shadow=.002,numPixels=1000,
                  n_clusters=10,region_of_interest=None):
    """
    Function that compute the cloud score given the differences between the real and predicted image.

    :param img_differences: image_real - image_pred
    :param threshold_dif_cloud: Threshold over the cloud score to be considered clouds
    :param threshold_dif_shadow:Threshold over the cloud score to be considered shadows
    :param n_clusters: number of clusters
    :param numPixels:  to be considered by the clustering algorithm
    :param region_of_interest:  region of interest within the image
    :return: ee.Image with 0 for clear pixels, 1 for shadow pixels and 2 for cloudy pixels
    """


    training = img_differences.sample(region=region_of_interest, scale=30, numPixels=numPixels)

    training, media, std = normalization.ComputeNormalizationFeatureCollection(training,
                                                                               BANDS_MODEL)

    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)
    img_differences_normalized = normalization.ApplyNormalizationImage(img_differences,
                                                                       BANDS_MODEL,
                                                                       media, std)
    result = img_differences_normalized.cluster(clusterer)

    cloud_score = SelectClusters(img_differences,result,n_clusters,region_of_interest)

    # Apply thresholds
    cloud_score_threshold = cloud_score.gt(threshold_dif_cloud).multiply(2)
    cloud_score_threshold = cloud_score_threshold.add(cloud_score.lt(-threshold_dif_shadow))

    # apply opening
    kernel = ee.Kernel.circle(radius=1)
    cloud_score_threshold = cloud_score_threshold.focal_min(kernel=kernel).\
        focal_max(kernel= kernel)

    # if no cloud no shadows
    any_cloudy_pixels = cloud_score_threshold.reduceRegion(ee.Reducer.sum(),
                                                           geometry=region_of_interest,
                                                           bestEffort=True,
                                                           scale=30).gt(0)

    return cloud_score_threshold.multiply(any_cloudy_pixels)






