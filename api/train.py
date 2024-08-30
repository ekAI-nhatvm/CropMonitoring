import sys
import os
from models.CNN_LSTM import NDVI_Reconstruction

#n_timesteps, n_features, n_outputs  = area_1.shape[1],area_1.shape[2],area_1.shape[1]
object_rec = NDVI_Reconstruction(n_timesteps=n_timesteps, n_features=n_features, n_outputs=n_outputs)
model = object_rec.config_model()
model.summary()