import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.io as sio
from scipy.io import loadmat, savemat
from tensorflow.keras import backend

def root_mean_squared_error(y_true, y_pred):
        return backend.sqrt(backend.mean(backend.square(y_pred-y_true)))
def mean_absolute_error(y_true, y_pred):
        return backend.mean(backend.abs(y_pred-y_true))

model_dtc_DPR_ocean = keras.models.load_model('Models/DPR/Ocean/model_dtc')
model_rtv_snow_DPR_ocean = keras.models.load_model('Models/DPR/Ocean/model_snow',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})

model_rtv_rain_DPR_ocean = keras.models.load_model('Models/DPR/Ocean/model_rain',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})

model_dtc_DPR_land = keras.models.load_model('Models/DPR/Land/model_dtc')
model_rtv_snow_DPR_land = keras.models.load_model('Models/DPR/Land/model_snow',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})
model_rtv_rain_DPR_land = keras.models.load_model('Models/DPR/Land/model_rain',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})

model_dtc_DPR_coast = keras.models.load_model('Models/DPR/Coast/model_dtc')
model_rtv_snow_DPR_coast = keras.models.load_model('Models/DPR/Coast/model_snow',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})
model_rtv_rain_DPR_coast = keras.models.load_model('Models/DPR/Coast/model_rain',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})

model_dtc_CPR_ocean = keras.models.load_model('Models/CPR/Ocean/model_dtc')
model_rtv_snow_CPR_ocean = keras.models.load_model('Models/CPR/Ocean/model_snow',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})
model_rtv_rain_CPR_ocean = keras.models.load_model('Models/CPR/Ocean/model_rain',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})

model_dtc_CPR_land = keras.models.load_model('Models/CPR/Land/model_dtc')
model_rtv_snow_CPR_land = keras.models.load_model('Models/CPR/Land/model_snow',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})

model_dtc_CPR_coast = keras.models.load_model('Models/CPR/Coast/model_dtc')
model_rtv_snow_CPR_coast = keras.models.load_model('Models/CPR/Coast/model_snow',
                                                  custom_objects={"root_mean_squared_error":root_mean_squared_error,
                                                                 "mean_absolute_error":mean_absolute_error})
#%%

files_DPR_ocean = sio.loadmat('Models/DPR/Ocean/files_DPR_ocean.mat')

mean0_detection_DPR_ocean = files_DPR_ocean['mean_detection_DPR_ocean']
std0_detection_DPR_ocean = files_DPR_ocean['std_detection_DPR_ocean']
mean0_snow_retrieval_DPR_ocean = files_DPR_ocean['mean_snow_retrieval_DPR_ocean']
std0_snow_retrieval_DPR_ocean = files_DPR_ocean['std_snow_retrieval_DPR_ocean']
mean0_rain_retrieval_DPR_ocean = files_DPR_ocean['mean_rain_retrieval_DPR_ocean']
std0_rain_retrieval_DPR_ocean = files_DPR_ocean['std_rain_retrieval_DPR_ocean']

mean_detection_DPR_ocean = np.transpose(mean0_detection_DPR_ocean)
std_detection_DPR_ocean = np.transpose(std0_detection_DPR_ocean)
mean_snow_retrieval_DPR_ocean = np.transpose(mean0_snow_retrieval_DPR_ocean)
std_snow_retrieval_DPR_ocean = np.transpose(std0_snow_retrieval_DPR_ocean)
mean_rain_retrieval_DPR_ocean = np.transpose(mean0_rain_retrieval_DPR_ocean)
std_rain_retrieval_DPR_ocean = np.transpose(std0_rain_retrieval_DPR_ocean)

files_DPR_land = sio.loadmat('Models/DPR/Land/files_DPR_land.mat')

mean0_detection_DPR_land = files_DPR_land['mean_detection_DPR_land']
std0_detection_DPR_land = files_DPR_land['std_detection_DPR_land']
mean0_snow_retrieval_DPR_land = files_DPR_land['mean_snow_retrieval_DPR_land']
std0_snow_retrieval_DPR_land = files_DPR_land['std_snow_retrieval_DPR_land']
mean0_rain_retrieval_DPR_land = files_DPR_land['mean_rain_retrieval_DPR_land']
std0_rain_retrieval_DPR_land = files_DPR_land['std_rain_retrieval_DPR_land']

mean_detection_DPR_land = np.transpose(mean0_detection_DPR_land)
std_detection_DPR_land = np.transpose(std0_detection_DPR_land)
mean_snow_retrieval_DPR_land = np.transpose(mean0_snow_retrieval_DPR_land)
std_snow_retrieval_DPR_land = np.transpose(std0_snow_retrieval_DPR_land)
mean_rain_retrieval_DPR_land = np.transpose(mean0_rain_retrieval_DPR_land)
std_rain_retrieval_DPR_land = np.transpose(std0_rain_retrieval_DPR_land)

files_DPR_coast = sio.loadmat('Models/DPR/Coast/files_DPR_coast.mat')

mean0_detection_DPR_coast = files_DPR_coast['mean_detection_DPR_coast']
std0_detection_DPR_coast = files_DPR_coast['std_detection_DPR_coast']
mean0_snow_retrieval_DPR_coast = files_DPR_coast['mean_snow_retrieval_DPR_coast']
std0_snow_retrieval_DPR_coast = files_DPR_coast['std_snow_retrieval_DPR_coast']
mean0_rain_retrieval_DPR_coast = files_DPR_coast['mean_rain_retrieval_DPR_coast']
std0_rain_retrieval_DPR_coast = files_DPR_coast['std_rain_retrieval_DPR_coast']

mean_detection_DPR_coast = np.transpose(mean0_detection_DPR_coast)
std_detection_DPR_coast = np.transpose(std0_detection_DPR_coast)
mean_snow_retrieval_DPR_coast = np.transpose(mean0_snow_retrieval_DPR_coast)
std_snow_retrieval_DPR_coast = np.transpose(std0_snow_retrieval_DPR_coast)
mean_rain_retrieval_DPR_coast = np.transpose(mean0_rain_retrieval_DPR_coast)
std_rain_retrieval_DPR_coast = np.transpose(std0_rain_retrieval_DPR_coast)

files_CPR_ocean = sio.loadmat('Models/CPR/Ocean/files_CPR_ocean.mat')

mean0_detection_CPR_ocean = files_CPR_ocean['mean_detection_CPR_ocean']
std0_detection_CPR_ocean = files_CPR_ocean['std_detection_CPR_ocean']
mean0_snow_retrieval_CPR_ocean = files_CPR_ocean['mean_snow_retrieval_CPR_ocean']
std0_snow_retrieval_CPR_ocean = files_CPR_ocean['std_snow_retrieval_CPR_ocean']

mean_detection_CPR_ocean = np.transpose(mean0_detection_CPR_ocean)
std_detection_CPR_ocean = np.transpose(std0_detection_CPR_ocean)
mean_snow_retrieval_CPR_ocean = np.transpose(mean0_snow_retrieval_CPR_ocean)
std_snow_retrieval_CPR_ocean = np.transpose(std0_snow_retrieval_CPR_ocean)


files_CPR_ocean_RR = sio.loadmat('Models/CPR/Ocean/files_CPR_ocean_RR.mat')

mean0_rain_retrieval_CPR_ocean = files_CPR_ocean_RR['mean_rain_retrieval_CPR_ocean_RR']
std0_rain_retrieval_CPR_ocean = files_CPR_ocean_RR['std_rain_retrieval_CPR_ocean_RR']

mean_rain_retrieval_CPR_ocean = np.transpose(mean0_rain_retrieval_CPR_ocean)
std_rain_retrieval_CPR_ocean = np.transpose(std0_rain_retrieval_CPR_ocean)

files_CPR_land = sio.loadmat('Models/CPR/Land/files_CPR_land.mat')

mean0_detection_CPR_land = files_CPR_land['mean_detection_CPR_land']
std0_detection_CPR_land = files_CPR_land['std_detection_CPR_land']
mean0_snow_retrieval_CPR_land = files_CPR_land['mean_snow_retrieval_CPR_land']
std0_snow_retrieval_CPR_land = files_CPR_land['std_snow_retrieval_CPR_land']

mean_detection_CPR_land = np.transpose(mean0_detection_CPR_land)
std_detection_CPR_land = np.transpose(std0_detection_CPR_land)
mean_snow_retrieval_CPR_land = np.transpose(mean0_snow_retrieval_CPR_land)
std_snow_retrieval_CPR_land = np.transpose(std0_snow_retrieval_CPR_land)


files_CPR_coast = sio.loadmat('Models/CPR/Coast/files_CPR_coast.mat')

mean0_detection_CPR_coast = files_CPR_coast['mean_detection_CPR_coast']
std0_detection_CPR_coast = files_CPR_coast['std_detection_CPR_coast']
mean0_snow_retrieval_CPR_coast = files_CPR_coast['mean_snow_retrieval_CPR_coast']
std0_snow_retrieval_CPR_coast = files_CPR_coast['std_snow_retrieval_CPR_coast']

mean_detection_CPR_coast = np.transpose(mean0_detection_CPR_coast)
std_detection_CPR_coast = np.transpose(std0_detection_CPR_coast)
mean_snow_retrieval_CPR_coast = np.transpose(mean0_snow_retrieval_CPR_coast)

std_snow_retrieval_CPR_coast = np.transpose(std0_snow_retrieval_CPR_coast)