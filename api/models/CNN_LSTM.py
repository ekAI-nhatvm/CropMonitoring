import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import tensorflow as tf
from tensorflow.keras import (
    Model,
    Sequential,
    losses,
    optimizers,
    metrics,
    layers,
    initializers,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.layers import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    LearningRateScheduler,
)
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import itertools
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint

class NDVI_Reconstruction:
    def __init__(self, n_timesteps, n_outputs, n_features):
        self.n_timesteps = n_timesteps
        self.n_outputs = n_outputs
        self.n_features = n_features

    def mse_custom(self, y_true, y_pred):
        err = tf.where(y_true > 0.0, y_true - y_pred, 0)
        return K.mean(K.square(err), axis=-1)

    def mae_custom(self, y_true, y_pred):
        err = tf.where(y_true > 0.0, y_true - y_pred, 0)
        return K.mean(K.abs(err), axis=-1)
    
    def add_CNN_block_1D(self, x_inp, filters, kernel_size=3, padding="same", strides=1):
        x = layers.Conv1D(filters,kernel_size,padding=padding, strides=strides,
                      kernel_initializer=initializers.glorot_normal())(x_inp)
        x = layers.Activation('relu')(x)
        return x
    
    def attention_seq(query_value, scale):
        query,value = query_value
        score = tf.matmul(query, value, transpose_b=True) #(batch, timestamp, 1)
        score = scale * score
        score = tf.nn.softmax(score, axis=1)
        score = score * query 
        
        return score
    
    def fusion_model(self, attention=False, cnn_layers=[8,16], pool_size=2, fcl_size=[16,16], lstm_units=32):
        inputs = list([])
        k = list([])
        var = features = np.array(['vv','vh','vv_div_vh','vh_minus_vv','ndvi'])
        
        for v in var:
            x_inp = layers.Input(shape=(self.n_timesteps, 1), name='{}_input'.format(v))
            inputs.append(x_inp)
            if v == 'ndvi':
                x_inp = layers.Masking(mask_value=-100)(x_inp)
            x = self.add_CNN_block_1D(x_inp,filters=cnn_layers[0])
            x = layers.Dropout(0.2)(x)
            for f in cnn_layers[1:]:
                x = self.add_CNN_block_1D(x, f)
                x = layers.Dropout(0.2)(x)
                
            x = layers.MaxPooling1D(pool_size=pool_size, strides=None)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Flatten()(x)
            for f in fcl_size[:-1]:
                x = layers.Dense(f,activation='relu',kernel_initializer=initializers.glorot_normal())(x)
            k.append(x)
        m = layers.Concatenate()(k)
        m = layers.RepeatVector(self.n_outputs)
        
        if attention:
            seq,state,_ = layers.LSTM(lstm_units, activation='relu',return_sequences=True,return_state=True)(m)
            att = tf.keras.layers.Lambda(self.attention_seq, arguments={'scale': 0.01})([seq, tf.expand_dims(state,1)])
            m = layers.LSTM(lstm_units, activation='relu',return_sequences=True)(att)
        else:
            m = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu'))(m)
            m = layers.RepeatVector(self.n_outputs)(m)
            m = layers.Bidirectional(layers.LSTM(lstm_units, activation='relu',return_sequences=True))(m)

        
        m = layers.TimeDistributed(layers.Dense(fcl_size[-1],activation='relu',kernel_initializer=initializers.glorot_normal()))(m)
        out = layers.TimeDistributed(layers.Dense(1,activation='sigmoid',kernel_initializer=initializers.glorot_normal()))(m)
        
        model = Model(inputs=inputs, outputs=out)
        
        return model
        
    def config_model(self):
        param_grid = {'cnn_layers': [[8,16]],
            'attention': [False],
            'pool_size':[3],
            'fcl_size':[[32,32]],
            'lstm_units':[16]}

        keys, values = zip(*param_grid.items())
        permutations_params = [dict(zip(keys, v)) for v in itertools.product(*values)]


        params = permutations_params[0]
        attention = params['attention']
        cnn_layers = params['cnn_layers']
        pool_size = params['pool_size']
        fcl_size = params['fcl_size']
        lstm_units = params['lstm_units']
    
    
        fusion_model = self.fusion_model(attention=attention, cnn_layers=cnn_layers, pool_size=pool_size, fcl_size=fcl_size, lstm_units=lstm_units)
        
        return fusion_model
    
    # Hàm scheduler để điều chỉnh learning rate
    def scheduler(self, epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.1

    # Sử dụng hàm scheduler với LearningRateScheduler
    change_lr = LearningRateScheduler(scheduler)    
    def training_model(self, X_new_train, y_train, X_new_val, y_val, masks_fused_train, masks_fused_val):
        fusion_model = self.config_model()
        fusion_model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])
        
        batch_size = 1024
        epochs = 100
        ver = 0
        initial_lr = 0.005
        
        model_path = 'models/fusion_model.weights.h5'
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=ver)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=ver, patience=7)
        change_lr = LearningRateScheduler(self.scheduler)
        mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', 
                            verbose=ver, save_best_only=True, save_weights_only=True)

        fusion_model.fit(X_new_train, y_train.reshape(-1, self.n_timesteps, 1),
                        validation_data=(X_new_val, y_val.reshape(-1, self.n_timesteps, 1), masks_fused_val),
                        batch_size=batch_size,
                        sample_weight=masks_fused_train,
                        epochs=epochs,
                        verbose=True,
                        callbacks=[mc, reduce_lr, es, change_lr]
                        )