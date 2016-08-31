# -*- coding: utf-8 -*-
"""
VGG- 16

Author: Some dude on github + Riaan

"""
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2, activity_l2
import os
import h5py


'''
VGG from: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

'''


def VGG_16(weights_path=None, DO = 0.5, shape = 32 ):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, shape,shape)))
    
    # previous size 32
    sz1 = 64
    model.add(Convolution2D(sz1, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz1, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    sz2 = 128
    model.add(ZeroPadding2D((1,1)))
    # previous size 128
    model.add(Convolution2D(sz2, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    sz3 = 256
    model.add(ZeroPadding2D((1,1)))
    # previous size 256
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    
    sz4=512
    # previous size 512
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # previous size 512
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(4, activation='softmax',
                    W_regularizer=l2(0.001), 
                    activity_regularizer=activity_l2(0.001)))
    
        
    return model

def VGG_16_gen(weights_path=None, DO = 0.5, shape = 224 ):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3, shape,shape)))
    
    # previous size 32
    sz1 = 64
    model.add(Convolution2D(sz1, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz1, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    sz2 = 128
    model.add(ZeroPadding2D((1,1)))
    # previous size 128
    model.add(Convolution2D(sz2, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))


    sz3 = 256
    model.add(ZeroPadding2D((1,1)))
    # previous size 256
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    
    sz4=512
    # previous size 512
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # previous size 512
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(1000, activation='softmax',
                    W_regularizer=l2(0.001), 
                    activity_regularizer=activity_l2(0.001)))
    
    # load the weights
    model.load_weights(weights_path)
    
    # pop last layer, insert my own
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    
    
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(4, activation='softmax',
                    W_regularizer=l2(0.001), 
                    activity_regularizer=activity_l2(0.001)))
    print 'Model Loaded'
    
        
    return model

def VGG_19(weights_path=None, DO =0.5, shape = 80):
    model = Sequential()
    
    sz1 = 64
    model.add(ZeroPadding2D((1,1),input_shape=(3, shape, shape)))
    model.add(Convolution2D(sz1, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz1, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    sz2 = 128
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz2, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz2, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    sz3 = 256
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz3, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    sz4=512
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(sz4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(DO))
    
    model.add(Dense(4, activation='softmax'))

    return model

'''
For testing purposes only!
'''
def VGG_16_test(weights_path='saved_models/best_model_VGG_16/weights_16.h5', shape = 112):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,shape,shape)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

# testing purposes only!
def VGG_19_test(weights_path='saved_models/best_model_VGG_19/weights_19.h5', shape = 112):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,shape,shape)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model