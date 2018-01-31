'''
Created on May 24, 2016

@author:  Gonzalo Mateo Garcia
@contact: gonzalo.mateo-garcia@uv.es

'''
import ee
import numpy as np
from ee_ipl_uv import converters


class RBFDistance:
    def __init__(self, gamma):
        self.gamma = gamma

    def arrayDistance(self, array_1, array_2):
        """
        Point to point distance

        :param array_1:
        :param array_2:
        :return:
        """
        array = array_1.subtract(array_2).pow(2)
        numerito = array.reduce(ee.Reducer.sum(), [0]).get([0])
        numerito = ee.Number(numerito).multiply(-self.gamma).exp()
        return numerito

    # def arrayImageNumpyDistance(self, array_image, numpy_kernel):
    #     """Returns array Image row vector with shape numpy_kernel.shape[0] x 1"""
    #     # For each pixel p compute the vector: ||p - x_i||^2
    #     # || p - x_i ||^2 = p^t*p + x_i^t*x_i -2*p^t*x_i
    #
    #     # x_i^t * x_i
    #     x_dist = [np.linalg.norm(numpy_kernel, axis=1).tolist()]
    #     x_dist = ee.Array(x_dist)  # 1 x numpy_kernel.shape[0]
    #
    #     # p^t * x_i
    #     matriz = ee.Array(numpy_kernel.T.tolist())  # numpy_kernel.shape[1] x numpy_kernel.shape[0]
    #     multiplicacion = array_image.matrixTranspose()  # 1 x numpy_kernel.shape[1]
    #     multiplicacion = multiplicacion.matrixMultiply(matriz)  # 1 x numpy_kernel.shape[0]
    #
    #     # p^t*p
    #     imagen_1_banda = array_image.matrixTranspose().matrixMultiply(array_image).arrayGet([0, 0])
    #
    #     # 2*p^t*x_i - p^t*p - x_i^t * x_i
    #     # 2* p^t*x_i
    #     menos_norma = multiplicacion.multiply(2)
    #     # 2*p^t*x_i - p^t*p
    #     menos_norma = menos_norma.subtract(imagen_1_banda)
    #     # 2*p^t*x_i - p^t*p - x_i^t * x_i
    #     menos_norma = menos_norma.subtract(x_dist)
    #
    #     # exp(- \gamma * || x_i -x ||) = exp(gamma * 2*p^t*x_i - p^t*p - x_i^t * x_i)
    #     return menos_norma.multiply(self.gamma).exp()  # numpy_kernel.shape[0] x 1

    def arrayImageeeArrayDistance(self, array_image1D, ee_array):
        """Returns array Image row vector with shape numpy_kernel.shape[0] x 1"""
        # For each pixel p compute the vector: ||p - x_i||^2
        # || p - x_i ||^2 = p^t*p + x_i^t*x_i -2*p^t*x_i

        # x_i^t * x_i
        x_dist = ee_array.multiply(ee_array).reduce(ee.Reducer.sum(), [1])
        x_dist = ee.Array(x_dist).matrixTranspose()  # 1 x numpy_kernel.shape[0]

        array_image2D = array_image1D.toArray(1)
        # p^t * x_i
        multiplicacion = array_image2D.matrixTranspose()  # 1 x numpy_kernel.shape[1]
        multiplicacion = multiplicacion.matrixMultiply(ee_array.matrixTranspose())  # 1 x numpy_kernel.shape[0]

        # p^t*p
        imagen_1_banda = array_image1D.arrayDotProduct(array_image1D)

        # 2*p^t*x_i - p^t*p - x_i^t * x_i
        # 2* p^t*x_i
        menos_norma = multiplicacion.multiply(2)
        # 2*p^t*x_i - p^t*p
        menos_norma = menos_norma.subtract(imagen_1_banda)
        # 2*p^t*x_i - p^t*p - x_i^t * x_i
        menos_norma = menos_norma.subtract(x_dist)

        # exp(- \gamma * || x_i -x ||) = exp(gamma * 2*p^t*x_i - p^t*p - x_i^t * x_i)
        return menos_norma.multiply(self.gamma).exp()  # 1 x numpy_kernel.shape[0]


class Kernel:
    """Class which implements a kernel defined by a feature collection and a distance"""
    def __init__(self, feature_collection, properties,
                 distancia=RBFDistance(.5), weight_property=None):

        assert type(properties) is list, \
            "properties should be a python list object"

        self.num_rows = ee.Number(feature_collection.size())

        # Remove weight propery if present
        if weight_property is not None:
            properties = list(filter(lambda prop: prop != weight_property, properties))
            self.weight_array = converters.eeFeatureCollectionToeeArray(feature_collection,
                                                                        [weight_property])
        else:
            self.weight_array = ee.Array(ee.List.repeat(1, self.num_rows))
            self.weight_array = ee.Array.cat([self.weight_array], 1)

        self.properties = properties

        assert len(self.properties) > 1, \
            "There is no properties in the current collection"

        # Get rid of extra columns
        feature_collection = feature_collection.select(properties)

        self.feature_collection = feature_collection

        self.kernel_numpy = None

        # We store in self.distancia the object which implement the distance
        self.distancia = distancia

        self.list_collection = feature_collection.toList(self.num_rows)

    def geteeArray(self):
        return converters.eeFeatureCollectionToeeArray(self.feature_collection,
                                                       self.properties)

    def getNumpy(self):
        if self.kernel_numpy is None:
            self.kernel_numpy = converters.eeFeatureCollectionToNumpy(self.feature_collection,
                                                                      self.properties)

        return self.kernel_numpy

    def getKeeArray(self):
        return ee.Array(self.applyToListCollection(self.list_collection))

    def getAlphaeeArray(self, array_y, lambda_regularization=0):
        """ Solve ridge regression. Returns alpha vector as ee.Array

        :param array_y: (ee.Array) y 2D array (self.num_rows x K)
        :param lambda_regularization: (double) regularization factor
        :return: (np.array) alpha array with shape(self.num_rows x K)
        :type: ee.Array
        """
        k_matrix = self.getKeeArray()
        if lambda_regularization > 0:
            k_matrix = k_matrix.add(self.weight_array.pow(-1).matrixToDiag().multiply(lambda_regularization))

        return k_matrix.matrixSolve(array_y)

    def getAlphaNumpy(self, array_y, lambda_regularization=0):
        """ Solve ridge regression. Returns alpha vector as np.array

        :param array_y: (ee.Array) y 2D array (self.num_rows x K)
        :param lambda_regularization: (double) regularization factor
        :return:  (np.array) numpy alpha array  with shape(self.num_rows x K)
        """
        alpha_server = self.getAlphaeeArray(array_y, lambda_regularization)
        return np.asanyarray(alpha_server.getInfo())

    def applyModelToImage(self, image, alpha):
        """ Apply ridge regression to image.

        :param image: (ee.Image) image object.
        :param alpha: (np.array|ee.Array|list) column vector
        :return: ee.Image
        """
        arrayImage1D = image.select(self.properties).toArray()

        return kernelMethodImage(arrayImage1D, self.geteeArray(), alpha,self.distancia)

    def applyToArray(self, array_feature):
        def distancia_features(feature):
            return self.distancia.arrayDistance(ee.Feature(feature).toArray(self.properties),
                                                array_feature)

        return self.list_collection.map(distancia_features)

    def applyToListCollection(self, list_col_2):
        def apply_to_feature_col_2(feature_a_aplicar):
            feature_a_aplicar = ee.Feature(feature_a_aplicar)
            array_feature = feature_a_aplicar.toArray(self.properties)
            # Get the function that compute the distance from a feature to this feature

            def distancia_features(feature):
                distancia = self.distancia.arrayDistance(ee.Feature(feature).toArray(self.properties),
                                                         array_feature)
                return distancia
            
            # Compute the distance from feature to every element in self.feature_collection
            return self.list_collection.map(distancia_features)

        return list_col_2.map(apply_to_feature_col_2)


def kernelMethodImage(arrayImage1D, inputs, alpha, distancia):
    """
    Function that applies a kernel method to every pixel of the image:
     if x is the pixel:

     f(x) = \sum_i distancia(x,inputs[i]) alpha[i]

    :param arrayImage1D:
    :param inputs:
    :param alpha:
    :param distancia:
    :return: an arrayImage of 1D with the product

    """

    if type(inputs) is  not ee.Array:
        if type(inputs) is np.ndarray:
            inputs = ee.Array(inputs.tolist())
        else:
            inputs = ee.Array(inputs)

    # Convert to Column vector
    array_image_fc = distancia.arrayImageeeArrayDistance(arrayImage1D,
                                                         inputs)

    if type(alpha) is ee.Array:
        alpha_server = alpha
    elif type(alpha) is np.ndarray:
        alpha_server = ee.Array(alpha.tolist())
    else:
        alpha_server = ee.Array(alpha)

    return array_image_fc.matrixMultiply(alpha_server)

