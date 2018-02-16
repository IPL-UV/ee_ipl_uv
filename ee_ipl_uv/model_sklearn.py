import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import expon
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)

def fit_model_local(ds_total,model,
                    bands_estimation_input,
                    bands_estimation_output,
                    center_output=True,
                    scale_termic_bands=True):

    # Shuffle data before fitting
    ds_total = ds_total.sample(frac=1).reset_index(drop=True)

    # Center and scale the output
    output_dataset = ds_total[bands_estimation_output].copy(deep=True)
    output_mean = np.zeros(output_dataset.shape[1],dtype=np.float64)
    if center_output:
        output_mean = output_dataset.mean()
        output_mean = output_mean[bands_estimation_output].values
        output_dataset -=  output_mean


    output_std = np.ones(output_dataset.shape[1],dtype=np.float64)
    if scale_termic_bands:
        if "B10" in bands_estimation_output:
            output_std[bands_estimation_output.index("B10")] = 100
        if "B11" in bands_estimation_output:
            output_std[bands_estimation_output.index("B11")] = 100

        output_dataset /= output_std

    # insert the weight depending on the model referenced
    if "weight" in ds_total:
        param_fit_params = [p for p in model.get_params().keys() if "fit_params" in p]
        if len(param_fit_params) > 0:
            dict_params = {param_fit_params[0]:
                               {"sample_weight": ds_total["weight"].values}}

            model.set_params(**dict_params)
            model.fit(ds_total[bands_estimation_input],
                      output_dataset)
        else:
            if hasattr(model, "named_steps"):
                model.fit(ds_total[bands_estimation_input],
                          output_dataset,
                          kernelridge__sample_weight=ds_total["weight"].values)
            else:
                model.fit(ds_total[bands_estimation_input],
                          output_dataset,
                          sample_weight=ds_total["weight"].values)

    else:
        model.fit(ds_total[bands_estimation_input],
                  output_dataset)

    y_hat = model.predict(ds_total[bands_estimation_input])

    # un-scale and un-center the output
    if scale_termic_bands:
        y_hat*= output_std

    if center_output:
        y_hat += output_mean

    y_hat = pd.DataFrame(y_hat, columns=bands_estimation_output)

    mae = mean_absolute_error(y_hat, ds_total[bands_estimation_output])
    mse = mean_squared_error(y_hat, ds_total[bands_estimation_output])
    logger.info("MAE: %.4f MSE: %.4f" % (mae, mse))

    return output_mean,output_std

def KRRModel(n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK",3)),
             cv=5, n_iter=30,verbose=2,best_params=None):

    if best_params is None:
        kr = RandomizedSearchCV(KernelRidge(kernel="rbf"),
                                param_distributions={"alpha": expon(scale=.02),
                                                     "gamma": expon(scale=.06)},
                                verbose=verbose,
                                n_jobs=n_jobs,
                                cv=cv,
                                n_iter=n_iter)
    else:
        if "kernel" not in best_params:
            best_params["kernel"] = "rbf"
        kr = KernelRidge(**best_params)

    model = make_pipeline(StandardScaler(),
                          kr)

    return model

def LinearModel(n_jobs=int(os.getenv("SLURM_CPUS_PER_TASK",3)),
                cv=5, verbose=1,best_params=None):
    if best_params is None:
        ridge_cv = GridSearchCV(Ridge(normalize=True),
                                verbose=verbose,
                                n_jobs=n_jobs,
                                cv=cv,
                                param_grid={"alpha": 10**np.arange(-5,10,1,dtype=np.float64)})
    else:
        if "normalize" not in best_params:
            best_params["normalize"] = True
        ridge_cv = Ridge(**best_params)

    return ridge_cv
